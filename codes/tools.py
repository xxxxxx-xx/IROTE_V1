
from abc import abstractmethod
import json
import os
import pandas as pd
from os import path
from typing import List, Tuple
from threading import Lock
import numpy as np
import re

# Try to import SimCSE (heavy dependency)
try:
    from sentence_transformers import SentenceTransformer, util
    import faiss
    import torch
    SIMCSE_AVAILABLE = True
except Exception as e:
    SIMCSE_AVAILABLE = False
    print(f"[Info] sentence_transformers/faiss not fully available: {e}")
    print("[Info] Retriever will use lightweight TF-IDF mode.")

def softmax(x, temperature=1.0):
    if len(x) == 0:
        return np.array([])
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)

class Retriever:
    """
    Retriever class for retrieving similar texts from a corpus.
    Supports two modes:
      - SimCSE mode: uses sentence-transformers + faiss (requires model download)
      - TF-IDF mode: uses sklearn TfidfVectorizer (lightweight, no download needed)
    """
    def __init__(self, model_name: str, corpus_texts: List[str], use_gpu: bool = False, save_path: str = None):
        self.corpus_texts = list(corpus_texts)
        self.lock = Lock()
        self.use_tfidf = not SIMCSE_AVAILABLE

        if self.use_tfidf:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._cosine_similarity = cosine_similarity
            self._vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            self.corpus_embeddings = self._vectorizer.fit_transform(self.corpus_texts)
        else:
            self.model = SentenceTransformer(model_name, device='cuda' if use_gpu else 'cpu')
            self.corpus_embeddings = self.model.encode(corpus_texts, batch_size=1024, show_progress_bar=False)
            if use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(self.corpus_embeddings.shape[1]))
            else:
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
            self.index.add(self.corpus_embeddings)

        if save_path:
            self.save_path = os.path.join(save_path, f"retriever.txt")
            self.save()

    def retrieve(self, query: str, top_k: int = 20, filter_texts: List[str] = []) -> List[Tuple[int, str, float]]:
        if self.use_tfidf:
            query_vec = self._vectorizer.transform([query])
            similarities = self._cosine_similarity(query_vec, self.corpus_embeddings).flatten()
            sorted_idx = np.argsort(similarities)[::-1]
            results = {'idx': [], 'text': [], 'similarity': []}
            for idx_i in sorted_idx:
                if len(results['idx']) >= top_k:
                    break
                text = self.corpus_texts[idx_i]
                if text in filter_texts:
                    continue
                results['idx'].append(idx_i)
                results['text'].append(text)
                results['similarity'].append(float(similarities[idx_i]))
            return results
        else:
            query_embedding = self.model.encode([query], show_progress_bar=False)[0]
            _, idx = self.index.search(query_embedding.reshape(1, -1), top_k + len(filter_texts))
            results = {'idx': [], 'text': [], 'similarity': []}
            for i in range(top_k):
                idx_i = idx[0][i]
                text = self.corpus_texts[idx_i]
                if text in filter_texts:
                    continue
                similarity = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings[idx_i]).item()
                results['idx'].append(idx_i)
                results['text'].append(text)
                results['similarity'].append(similarity)
            return results

    def add_corpus(self, new_corpus: List[str]):
        self.corpus_texts += new_corpus
        if self.use_tfidf:
            self.corpus_embeddings = self._vectorizer.fit_transform(self.corpus_texts)
        else:
            new_corpus_embeddings = self.model.encode(new_corpus, batch_size=1024, show_progress_bar=False)
            self.corpus_embeddings = np.vstack([self.corpus_embeddings, new_corpus_embeddings])
            self.index.add(new_corpus_embeddings)
        self.save()

    def dedup_and_add_corpus(self, new_corpus: List[str], threshold=0.85):
        if self.use_tfidf:
            new_vecs = self._vectorizer.transform(new_corpus)
            sim_matrix = self._cosine_similarity(new_vecs, self.corpus_embeddings)
            deduped_texts = []
            for i in range(len(new_corpus)):
                if sim_matrix[i].max() <= threshold:
                    deduped_texts.append(new_corpus[i])
        else:
            new_corpus_embeddings = self.model.encode(new_corpus, show_progress_bar=False, batch_size=1024)
            similarity_matrix = util.pytorch_cos_sim(new_corpus_embeddings, self.corpus_embeddings)
            mask = similarity_matrix > threshold
            deduped_texts = []
            for i in range(len(new_corpus)):
                if not mask[i].any():
                    deduped_texts.append(new_corpus[i])
        if len(deduped_texts) > 0:
            self.add_corpus(deduped_texts)

    def calculate_similarity(self, query: str, texts: List[str]) -> np.array:
        if len(texts) == 0:
            return np.array([])
        if self.use_tfidf:
            query_vec = self._vectorizer.transform([query])
            text_vecs = self._vectorizer.transform(texts)
            return self._cosine_similarity(query_vec, text_vecs).flatten()
        else:
            query_embedding = self.model.encode([query], show_progress_bar=False)[0]
            text_embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=1024)
            similarity_scores = util.pytorch_cos_sim(query_embedding, text_embeddings)
            return similarity_scores[0].cpu().numpy()

    def calculate_max_similarity(self, query: str, texts: List[str]) -> float:
        similarity_scores = self.calculate_similarity(query, texts)
        return np.max(similarity_scores) if len(similarity_scores) > 0 else 0

    def calculated_weighted_similarity(self, similarities: List[float], temperature: float =1.0):
        scores = np.array(similarities)
        weights = softmax(scores, temperature)
        weighted_sum = np.sum(weights * scores)
        return weighted_sum

    def calculate_texts_mean_distance(self, texts: str):
        if self.use_tfidf:
            text_vecs = self._vectorizer.transform(texts)
            similarity_matrix = self._cosine_similarity(text_vecs, text_vecs)
            np.fill_diagonal(similarity_matrix, 0)
            return 1 - similarity_matrix.mean()
        else:
            text_embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=1024)
            similarity_matrix = util.pytorch_cos_sim(text_embeddings, text_embeddings)
            mask = torch.eye(len(texts)).bool()
            similarity_matrix[mask] = 0
            mean_distance = 1 - similarity_matrix.mean().item()
            return mean_distance

    def calculate_out_cluster_distance(self, in_cluster_texts: str):
        sum_distance = 0
        for text in in_cluster_texts:
            res = self.retrieve(text, top_k=len(in_cluster_texts) - 1, filter_texts=in_cluster_texts)
            sum_distance += (1 - np.mean(res['similarity']))
        return sum_distance / len(in_cluster_texts)

    def save(self, save_path: str = None):
        if hasattr(self, 'save_path') and save_path is None:
            with open(self.save_path, 'w') as f:
                f.write('\n'.join(self.corpus_texts))
        elif save_path:
            with open(save_path, 'w') as f:
                f.write('\n'.join(self.corpus_texts))

    @classmethod
    def load_corpus(cls, save_path: str):
        save_path = os.path.join(save_path, "retriever.txt")
        with open(save_path, 'r') as f:
            corpus_texts = [text.strip() for text in f.readlines()]
        return corpus_texts



def extract_quest(text):
    text = text.strip().strip('"')
    if not isinstance(text, str):
        return None
    questions = re.findall(r'\[Question\]: (.*?)(?:\n|\[|$)', text)
    if len(questions) > 0:
        return questions[0]
    elif  text.endswith("?"):
        if ":" in text or "：" in text:
            return text.split(":", 1)[1].strip() if ":" in text else text.split("：", 1)[1].strip()
        else:
            return text
    else:
        return None 

def extract_arg_quest(text): # _stance_statement
    if not isinstance(text, str):
        return None, None
    # use regex to extract specific patterns
    arguments = re.findall(r'\[Argument\]: (.*?)(?:\n|\[|$)', text)
    questions = re.findall(r'\[Question\]: (.*?)(?:\n|\[|$)', text)
    len_args, len_ques = len(arguments), len(questions)
    if 0 in [len_args, len_ques]:
        # retrying
        arguments = re.findall(r'\S*\s*Argument\s*\S*: (.*?)(?:\n|\[|$)', text)
        questions = re.findall(r'\S*\s*Question\s*\S*: (.*?)(?:\n|\[|$)', text)
        len_args, len_ques = len(arguments), len(questions)
        # still failed
        if 0 in [len_args, len_ques]:
            # last try
            arguments = re.findall(r'[^:]*Argument[^:]*: (.*?)(?:\n|\[|$)', text)
            questions = re.findall(r'[^:]*Question[^:]*: (.*?)(?:\n|\[|$)', text)
            len_args, len_ques = len(arguments), len(questions)
            if 0 in [len_args, len_ques]:
                return None, None
    if not 0 in [len_args, len_ques]:
        return arguments[0], questions[0]
    else:
        return None, None

def split_and_extract(text):
    # use regex to split text, keep the splitters
    if '###' in text:
        split_list = re.split(r'(?=###)', text)
    elif 'Case' in text:
        split_list = re.split(r'(?=Case \d:)', text)
    else:
        pattern = r'^\d+\.\s*'
        split_list = re.split(pattern, text, flags=re.MULTILINE)
    arguments, stances, statements, questions = [], [], [], []
    for split in split_list:
        out = extract_args_quests_stances_statements(split)
        if  None not in out:
            argument, stance, statement, question = out
            if len(argument) == len(stance) == len(statement) == len(question):
                arguments.extend(argument)
                stances.extend(stance)
                statements.extend(statement)
                questions.extend(question)
    return arguments, stances, statements, questions

def extract_args_quests_stances_statements(text):
    # use regex to extract specific patterns
    arguments = re.findall(r'\[Argument\]: (.*?)(?:\n|\[|$)', text)
    questions = re.findall(r'\[Question\]: (.*?)(?:\n|\[|$)', text)
    stances = re.findall(r'\[Stance\]: (.*?)(?:\n|\[|$)', text)
    statements = re.findall(r'\[Statement\]: (.*?)(?:\n|\[|$)', text)
    len_args, len_ques, len_stan, len_stat = len(arguments), len(questions), len(stances), len(statements)
    if 0 in [len_args, len_ques, len_stan, len_stat]:
        # retrying
        arguments = re.findall(r'\S*\s*Argument\s*\S*: (.*?)(?:\n|\[|$)', text)
        questions = re.findall(r'\S*\s*Question\s*\S*: (.*?)(?:\n|\[|$)', text)
        stances = re.findall(r'\S*\s*Stance\s*\S*: (.*?)(?:\n|\[|$)', text)
        statements = re.findall(r'\S*\s*Statement\s*\S*: (.*?)(?:\n|\[|$)', text)
        len_args, len_ques, len_stan, len_stat = len(arguments), len(questions), len(stances), len(statements)
        # still failed
        if 0 in [len_args, len_ques, len_stan, len_stat]:
            # last try
            arguments = re.findall(r'[^:]*Argument[^:]*: (.*?)(?:\n|\[|$)', text)
            questions = re.findall(r'[^:]*Question[^:]*: (.*?)(?:\n|\[|$)', text)
            stances = re.findall(r'[^:]*Stance[^:]*: (.*?)(?:\n|\[|$)', text)
            statements = re.findall(r'[^:]*Statement[^:]*: (.*?)(?:\n|\[|$)', text)
            len_args, len_ques, len_stan, len_stat = len(arguments), len(questions), len(stances), len(statements)
            if 0 in [len_args, len_ques, len_stan, len_stat]:
                return None, None, None, None
    if len(arguments) == len(questions) == len(stances) == len(statements):
        return arguments, stances, statements, questions
    else:
        return None, None, None, None

def extract_stance(text):
    if not isinstance(text, str):
        return None
    stance = re.findall(r'\[Stance\]:\s(.*)', text)
    if len(stance) > 0:
        text = stance[0]
    if 'in favor' in text.lower():
        stance = 'in favor of'
    elif 'against' in text.lower():
        stance = 'against'
    elif 'neutral' in text.lower() or "cautiously" in text.lower() or "cautious" in text.lower() or "balanced" in text.lower():
        stance = 'neutral'
    else:
        stance = None
    return stance

def english_positive_words(text):
    return 'yes' in text.lower() or 'in favor' in text.lower() or 'support' in text.lower() or 'agree' in text.lower()

def english_negative_words(text):
    return 'no' in text.lower() or 'against' in text.lower() or 'oppose' in text.lower() or 'disagree' in text.lower()

def chinese_positive_words(text):
    return '支持' in text or '赞成' in text or '同意' in text

def chinese_negative_words(text):
    return '反对' in text or '不支持' in text or '不同意' in text

def extract_generation_stance(text):
    if not isinstance(text, str):
        return 0
    if english_negative_words(text) or chinese_negative_words(text):
        return -1
    elif english_positive_words(text) or chinese_positive_words(text):
        return 1
    else:
        return 0

def extract_generation_evidences(text):
    """
    Stance: <yes/no/neutral>
    Key Points:
    1. {{first point}}: {{justification}}
    2. {{second point}}: {{justification}}
    3. {{third point}}: {{justification}}
    """
    if not isinstance(text, str) or 'Key Points' not in text:
        return None, None, None, None
    stance_text, key_points_text = text.split('Key Points:', 1) if 'Key Points:' in text else text.split('Key Points', 1)
    # first split: a. b. c.
    stance = stance_text.replace('Stance:', '').strip()
    evidences = re.split(r'[1-3]\.', key_points_text.strip())
    key_points, justifications, validate_evidences = [], [], []
    for evidence in evidences:
        if len(evidence) == 0 or ':' not in evidence:
            continue
        point, justification = evidence.split(':', 1)
        # remove markdown syntax like **, *, etc.
        point = re.sub(r'\*+', '', point).strip()
        justification = re.sub(r'\*+', '', justification).strip()
        validate_evidences.append(evidence.strip())
        key_points.append(point)
        justifications.append(justification)
    return stance, key_points, justifications, validate_evidences

# extract score from -10 to 10
def extract_score_annotation(text):
    # extract score from -10 to 10
    if not isinstance(text, str):
        return None
    score = re.findall(r'(-?\d)', text)
    if len(score) == 0:
        return 0
    return int(score[0])

class GenerationText:
    def __init__(self, model_name, raw_text, stance_text=None, key_points_text=None, evidence_text=None, value_list=None):
        self.model_name = model_name
        self.raw_text = raw_text
        self.stance_text = stance_text
        self.key_points = key_points_text
        self.evidence_text = evidence_text
        self.value_list = value_list
    
    def __dict__(self):
        return {
            'model_name': self.model_name,
            'stance_text': self.stance_text,
            'key_points_text': self.key_points,
            'evidence_text': self.evidence_text,
            'value_list': self.value_list,
            'raw_text': self.raw_text
        }

class Argument:
    def __init__(self, argument, 
                 question, 
                 total_rewards=None, 
                 final_total_rewards=None, 
                 reward_info=None, 
                 final_reward_info=None, 
                 generation_texts=None, 
                 final_generation_texts=None):
        # generation_texts : List[GenerationText]
        self.argument = argument
        self.question = question
        self.generation_texts = generation_texts
        self.reward_info = reward_info
        self.total_rewards = total_rewards
        self.final_total_rewards = final_total_rewards
        self.final_reward_info = final_reward_info
        self.final_generation_texts = final_generation_texts

    def update_final_information(self, final_total_rewards, final_reward_info, final_generation_texts):
        self.final_total_rewards = final_total_rewards
        self.final_reward_info = final_reward_info
        self.final_generation_texts = final_generation_texts
    
    def __str__(self):
        return self.argument
    
    def __repr__(self) -> str:
        return self.argument
    
    def __dict__(self):
        return {
            'argument': self.argument,
            'total_rewards': self.total_rewards,
            'question': self.question,
            'final_total_rewards': self.final_total_rewards,
            'generation_texts': json.dumps([generation_text.__dict__() for generation_text in self.generation_texts]) if self.generation_texts else None,
            'reward_info': json.dumps(self.reward_info) if self.reward_info is not None else None,
            'final_reward_info': json.dumps(self.final_reward_info) if self.final_reward_info is not None else None,
            'final_generation_texts': json.dumps([generation_text.__dict__() for generation_text in self.final_generation_texts]) if self.final_generation_texts else None
        }
    
    @classmethod
    def load_gen_texts(cls, j_strings):
        try:
            gen_list = json.loads(j_strings)
            return [GenerationText(**generation_text) for generation_text in gen_list]
        except Exception as e:
            print(e)
            if isinstance(j_strings, str):
                print(j_strings[-50:])
            else:
                print(j_strings)
            return []

    @classmethod
    def from_dict(cls, d):
        argument = d['argument']
        question = d['question']
        total_rewards = d['total_rewards'] if isinstance(d['total_rewards'], float) else eval(d['total_rewards'])[0]
        final_total_rewards = d['final_total_rewards']
        generation_texts = cls.load_gen_texts(d['generation_texts']) # [GenerationText(**generation_text) for generation_text in json.loads(d['generation_texts'])] if isinstance(d['generation_texts'], str) else None
        reward_info = json.loads(d['reward_info']) if isinstance(d['reward_info'], str) else None
        final_reward_info = json.loads(d['final_reward_info']) if isinstance(d['final_reward_info'], str) else None
        final_generation_texts = cls.load_gen_texts(d['final_generation_texts']) 
        
        return Argument(argument=argument, 
                        question=question, 
                        total_rewards=total_rewards, 
                        final_total_rewards=final_total_rewards, 
                        reward_info=reward_info, 
                        final_reward_info=final_reward_info, 
                        generation_texts=generation_texts, 
                        final_generation_texts=final_generation_texts)

    def to_generation_df(self):
        if not self.generation_texts or len(self.generation_texts) == 0:
            return pd.DataFrame()
        tmp = pd.DataFrame([generation_text.__dict__() for generation_text in self.generation_texts if generation_text is not None])
        tmp['argument'] = self.argument
        tmp['question'] = self.question
        tmp['gen_type'] = 'eval'

        final_tmp = pd.DataFrame([generation_text.__dict__() for generation_text in self.final_generation_texts if generation_text is not None])
        final_tmp['argument'] = self.argument
        final_tmp['question'] = self.question
        final_tmp['gen_type'] = 'final'
        res = pd.concat([tmp, final_tmp], axis=0)
        return res
    
    def to_argument_df(self):
        return pd.DataFrame([self.__dict__()])
    
    def to_reward_dict(self):
        return {
            'argument': self.argument,
        }.update(self.reward_info)
    
    def __compare__(self, other):
        # from high to low
        return self.total_rewards - other.total_rewards
    
    def __lt__(self, other):
        return self.total_rewards < other.total_rewards


class ValueEvaluator:
    def __init__(self, number_of_generations):
        self.number_of_generations = number_of_generations

    """Base class for value evaluator"""
    def diversity_score(self, general_argument: str, model_generations: List[List[List[str]]]) -> dict:
        """
        input: 
        1. general_argument: a general argument
        2. model_generations: [[model1_gen_list1, model1_gen_list2, ...], [model2_gen_list1, model2_gen_list2, ...]]
        predict values and calculate diversity score
        1. internal value cohenrence: len(union(text_values)) / len(intersection(text_values))
        2. model value diversity: len(union(model_values)) / len(model_values)
        3. general_argument value diversity: avg(l1 distance between general_argument value and model_values)
        """
        model2text_idxs = {}
        orginize_texts = [general_argument]
        idx = 1
        # step 1: orginize texts
        number_of_models = len(model_generations)
        number_of_generations = self.number_of_generations
        for model_idx, model_gens_list in enumerate(model_generations):
            tmp_model_gen2_idx = {i: [] for i in range(number_of_generations)}
            for i, model_gens in enumerate(model_gens_list):
                # i th generation of model_idx: [evidence1, evidence2, ...]
                orginize_texts.extend(model_gens)
                tmp_model_gen2_idx[i] = list(range(idx, idx + len(model_gens)))
                idx += len(model_gens)
            model2text_idxs[model_idx] = tmp_model_gen2_idx
        # step 2: predict values
        # print(f"value pred: {len(orginize_texts)} texts")
        predct_res = self.predict_batch(orginize_texts)
        text_values = predct_res['text_values']
        value_dicts = predct_res['value_dict']
        general_argument_values = text_values[0]
        cohenrence, general_argument_distance = [], []
        # step 3: reorginize values by different generations
        # [gen1, gen2, gen3, ...]
        model_values = [set() for _ in range(number_of_models)]
        model_text_labels = [[[] for _ in range(number_of_models)] for _ in range(number_of_generations)]
        model_value_dicts = [[[] for _ in range(number_of_models)] for _ in range(number_of_generations)]
        for model_idx, gen2_idx in model2text_idxs.items():
            for model_gen_idx, model_text_idxs in gen2_idx.items():
                tmp_model_values_labels = [text_values[i] for i in model_text_idxs]
                tmp_model_values_dicts = [value_dicts[i] for i in model_text_idxs]
                coherence_score = self.calculate_diversity_score(tmp_model_values_labels)
                cohenrence.append(coherence_score)

                generation_unique_model_values = list(set([value for values in tmp_model_values_labels for value in values]))
                general_argument_distance.append(self.calculate_l1_distance(general_argument_values, generation_unique_model_values))
                model_values[model_idx].update(generation_unique_model_values)
                model_text_labels[model_gen_idx][model_idx] = tmp_model_values_labels
                model_value_dicts[model_gen_idx][model_idx] = tmp_model_values_dicts
            
        coherence_score = np.mean(cohenrence)
        general_argument_distance = np.mean(general_argument_distance)
        model_diversity = self.calculate_diversity_score(model_values)

        return {
            'context_coherence': coherence_score,
            'value_model_diversity': model_diversity,
            'topic_reguliarization': general_argument_distance, # larger the distance, the better for question
            'topic_values': general_argument_values,
            'model_values': [list(values) for values in model_values],
            'model_evidence_values': model_text_labels,
            'model_value_dicts': model_value_dicts
        }
               

    def calculate_diversity_score(self, text_values: List[List[str]]) -> float:
        """
        calculate diversity score based on text values
        """
        union_text_values, intersection_text_values = set(), set()
        for values in text_values:
            if len(intersection_text_values) == 0:
                intersection_text_values.update(values)
            union_text_values.update(values)
            intersection_text_values.intersection_update(values)
        internal_value_coherence = len(union_text_values) / len(intersection_text_values) if len(intersection_text_values) > 0 else len(union_text_values)
        return internal_value_coherence
    
    def calculate_l1_distance(self, value1: List[str], value2: List[str]) -> float:
        """
        calculate l1 distance between two value lists
        """
        # return len(set(value1) - set(value2))
        return len(set(value1) - set(value2)) + len(set(value2) - set(value1)) # bidirectional distance

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[List[str]]:
        pass

def check_schwartz_values(text):
    schwartz_values = [
        'Achievement',
        'Benevolence',
        'Conformity',
        'Hedonism',
        'Power',
        'Security',
        'Self-Direction',
        'Stimulation',
        'Tradition',
        'Universalism'
    ]
    for value in schwartz_values:
        if value.lower() in text.lower():
            return value
    return None

def extract_value_score(text, threshold=7.5):
    if not isinstance(text, str):
        return None
    text = text.strip()
    # first split: a. b. c.
    texts = re.split(r'[a-z]\.', text)
    if len(texts) == 0:
        texts = text.split('\n')
    # extract score int
    ret_value, ret_value_dict = set(), {}
    for t in texts:
        value = check_schwartz_values(t)
        if value:
            score = re.findall(r'\d+', t)
            if score:
                score = int(score[0])
                if score >= threshold:
                    ret_value.add(value)
                ret_value_dict[value] = score
    return list(ret_value), ret_value_dict

def get_text_value_prompt(text: str) -> str:
    """Generate a prompt for value prediction from text"""
    schwartz_values = [
        'Achievement', 'Benevolence', 'Conformity', 'Hedonism', 'Power',
        'Security', 'Self-Direction', 'Stimulation', 'Tradition', 'Universalism'
    ]
    values_str = ", ".join(schwartz_values)
    return f"""Analyze the following text and identify which Schwartz values it reflects.
Values: {values_str}

Text: {text}

For each value, provide a score from 0 to 10 indicating how strongly the text reflects that value.
Format: a. <Value>: <score>
b. <Value>: <score>
..."""


class LLMEvaluator(ValueEvaluator):
    """Use LLM to predict values"""
    def __init__(self,  model_name, model_router, number_of_generations, value_threshold=7.5):
        super().__init__(number_of_generations)
        self.model_name = model_name
        self.model_router = model_router
        self.value_threshold = value_threshold

    def predict_batch(self, texts: List[str]) -> List[List[str]]:
        """
        predict values for a batch of texts
        """
        prompts = [get_text_value_prompt(text) for text in texts]
        messages = [[{"role": "user", "content": p}] for p in prompts]
        pred_texts = self.model_router.request_llm(messages, model=self.model_name, max_length=512, temperature=0.0)
        text_values, value_dicts = [], []
        for text in pred_texts:
            value_list, value_dict = extract_value_score(text, self.value_threshold)
            text_values.append(value_list)
            value_dicts.append(value_dict)
        return {
            'text_values': text_values,
            'value_dict': value_dicts
        }

class VLLMEvaluator(ValueEvaluator):
    
    def __init__(self, router, number_of_generations):
        super().__init__(number_of_generations)
        self.router = router
        self.values = [
            'Achievement',
            'Benevolence',
            'Conformity',
            'Hedonism',
            'Power',
            'Security',
            'Self-direction',
            'Stimulation',
            'Tradition',
            'Universalism'
        ]
        self.eval_model_name = 'GPT-4o-Mini'
    
    def get_prompted_text(self, text, value):
        raw_instruct_value_instruct = """For the following task, you can reference the following list of Schwartz values and their definitions:
1. Self-direction - independent thought and action—choosing, creating, exploring
2. Stimulation - excitement, novelty and challenge in life
3. Hedonism - pleasure or sensuous gratification for oneself
4. Achievement - personal success through demonstrating competence according to social standards
5. Power - social status and prestige, control or dominance over people and resources
6. Security - safety, harmony, and stability of society, relationships, and of self
7. Conformity - restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms
8. Tradition - respect, commitment, and acceptance of the customs and ideas that one's culture or religion provides
9. Benevolence - preserving and enhancing the welfare of those with whom one is in frequent personal contact (the 'in-group')
10. Universalism - understanding, appreciation, tolerance, and protection for the welfare of all people and for nature

You are an AI assistant tasked with annotating whether a text reflects a specific Schwartz value. You will be presented with a text and a Schwartz value, and you should output whether the text reflects the given value. Just respond with 'Yes' or 'No'."""

        return raw_instruct_value_instruct + f"""\n### [Input]: {text}
### [Value]：{value}
### [Label]:"""

    
    def predict_batch(self, texts: List[str]) -> List[List[str]]:
        # prepare prompts
        prompts = []
        for text in texts:
            for value in self.values:
                prompts.append(self.get_prompted_text(text, value))
        # predict
        messages = [[{"role": "user", "content": p}] for p in prompts]
        pred_texts = self.router.request_llm(messages, model=self.eval_model_name, max_length=100, temperature=0.0)
        # print(f"pred_texts: {len(pred_texts)}")
        # print(f"pred_texts: {pred_texts[0]}")
        def transform_yes_no(x):
            if 'yes' in x.lower():
                return 1
            elif 'no' in x.lower():
                return 0
            else:
                return 0
        text_values, value_dicts = [], []
        for i in range(0, len(pred_texts), len(self.values)):
            values = [transform_yes_no(x) for x in pred_texts[i:i+len(self.values)]]
            text_values.append([self.values[i] for i, x in enumerate(values) if x == 1])
            value_dicts.append({self.values[i]: values[i] for i in range(len(self.values))})
        return {
            'text_values': text_values,
            'value_dict': value_dicts
        }        

class HuggingfaceEvaluator(ValueEvaluator):
    """Use a classifier to predict values"""
    def __init__(self, number_of_generations, device='cuda', bath_size=128):
        super().__init__(number_of_generations)
        from config import CLASSIFIER_PATH
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  
        self.tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)  
        self.model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH).half().to(device)
        self.model.eval()
        self.device = device
        self.bath_size = bath_size
        self.value_list = [
            'Achievement',
            'Benevolence',
            'Conformity',
            'Hedonism',
            'Power',
            'Security',
            'Self-Direction',
            'Stimulation',
            'Tradition',
            'Universalism'
        ]

    def predict_batch(self, texts: List[str]) -> List[List[str]]:
        """
        predict values for a batch of texts
        """
        text_len = len(texts)
        null_idx, input_texts = [], []
        idx = 0
        for text in texts:
            for value in self.value_list:
                if isinstance(text, str):
                    input_texts.append((value, text))
                else:
                    null_idx.append(idx)
                    input_texts.append((value, ''))
                idx += 1

        len_combination = len(input_texts)
        final_preds, final_probs = [], []
        with torch.no_grad():
            for i in range(0, len_combination, self.bath_size):
                batch = input_texts[i:i+self.bath_size] if i + self.bath_size < len_combination else input_texts[i:]
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                prob = torch.softmax(logits, dim=1).cpu()[:, 1].tolist()
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                final_preds.extend(preds)
                final_probs.extend(prob)
        
        text_values = [[] for _ in range(text_len)]
        text_dicts = [{} for _ in range(text_len)]
        for i, pred in enumerate(final_preds):
            text_idx = i // len(self.value_list)
            value_idx = i % len(self.value_list)
            if i in null_idx:
                text_dicts[text_idx][self.value_list[value_idx]] = 0
                continue
            if pred == 1:
                text_values[text_idx].append(self.value_list[value_idx])
            text_dicts[text_idx][self.value_list[value_idx]] = final_probs[i]
        return {
            'text_values': text_values,
            'value_dict': text_dicts
        }

def merge_results(results_list, avg_mode='algorithmic_mean'):
    """
    Merge the results from different questionnaires
    """
    res_dict = {}
    for res in results_list:
            for k, v in res.items():
                if k in res_dict:
                    res_dict[k].append(v)
                else:
                    res_dict[k] = [v]
    if avg_mode == 'algorithmic_mean':
        for k, v in res_dict.items():
            res_dict[k] = sum(v) / len(v)
    elif avg_mode == 'harmonic_mean':
        for k, v in res_dict.items():
            res_dict[k] = len(v) / sum([1 / val for val in v])
    elif avg_mode == 'geometric_mean':
        from functools import reduce
        for k, v in res_dict.items():
            res_dict[k] = (reduce(lambda x, y: x*y, v))**(1/len(v))
    elif avg_mode == 'max':
        for k, v in res_dict.items():
            res_dict[k] = max(v)
    elif avg_mode == 'min':
        for k, v in res_dict.items():
            res_dict[k] = min(v)
    # rescale to 0 - 10
    return res_dict