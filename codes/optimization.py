import random
from typing import List, Tuple
random.seed(312)
import os
import re
import json
import pandas as pd
from eval_utils import Evaluator
import numpy as np
from eval_utils.survey_handler import *
from probability_estimator import calculate_compactness
from tools import Retriever
import argparse
from tools import merge_results
from models.router import ModelRouter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from eval_utils.controller import Controller
from prompts import cot_optimize_reflections, optimize_reflections, summary_reflections
from config import PATH_TO_SIMCSE_MODEL
def extract_reflection(response):
    # extract 1. xxxxx \n 2. xxxxx \n 3. xxxxx
    return re.findall(r'\d+\.\s(.+)', response)

class ReflectionSet:
    """
    Storing a set of reflections.
    This class manages these reflections, including shuffling their order, recording evaluation results, and computing the average reward.
    """
    def __init__(self, strs, num_item_shuffle: int, iteration: int = 0):
        self.strs = strs
        n = len(strs)
        def get_num_possible_shuffle(n):
            num_possible_shuffle = 1
            for i in range(1, n+1):
                num_possible_shuffle *= i
            return num_possible_shuffle
        # number of shuffles
        assert num_item_shuffle <= get_num_possible_shuffle(n)
        # shuffle to get different indexs
        self.shuffle_indexs = [[i for i in range(n)]]
        if num_item_shuffle >= 1:
            for _ in range(num_item_shuffle):
                random.shuffle(self.shuffle_indexs[-1])
                if self.shuffle_indexs[-1] not in self.shuffle_indexs:
                    self.shuffle_indexs.append(self.shuffle_indexs[-1])

        self.record = [] # (index, text, score)
        # all_dimention status
        self.all_record = [] # {str: {dim1: xx, dim2: xx}...}
        self.avg_reward = None
        self.compactness = None
        self.term1 = None
        self.term2 = None
        self.iteration = iteration

    def get_integrated_strs(self, str_list = None)->str:
        """
        Get integrated strs By STRS
        [reflection1, reflection2, reflection3] -> 1. reflection1 \n 2. reflection2 \n 3. reflection3
        """
        if str_list is None:
            str_list = self.strs
        return "\n".join([f"{i+1}. {s}" for i, s in enumerate(str_list)])

    def __str__(self)->str:
        return self.get_integrated_strs(self.strs)
    
    def get_strs_from_indexs(self, indexs)->List[str]:
        return [self.strs[idx] for idx in indexs]
    
    def get_query_strs(self)->List[Tuple[int, str]]:
        """
        Return [(indexs, strs), ...]
        1. xxxxx
        2. xxxxx
        3. xxxxx
        """
        res_list = []
        # print('shuffle_indexs', self.shuffle_indexs)
        for indexs in self.shuffle_indexs:
            strs_with_index = self.get_strs_from_indexs(indexs)
            res_list.append((indexs, self.get_integrated_strs(strs_with_index)))
        return res_list
    
    def get_raw_strs(self)->List[str]:
        return self.strs
    
    def add_eval_record(self, index, text, score, all_dimention=None):
        """
        Add an evaluation record.

        Args:
            index: The index of the reflection in the current evaluation order.
            text: The concatenated string used during evaluation.
            score: The evaluation score for this set of reflections.
            all_dimention: Optional; a dictionary containing the evaluation results for all dimensions.
        """
        if all_dimention is not None:
            self.all_record.append(all_dimention)
        self.record.append((index, text, score))
        # print('record added', self.record)

    def get_reward(self)->float:
        """
        Calculate and return the average reward score of the current ReflectionSet.
        """
        self.avg_reward = np.mean([score for _, _, score in self.record])

        # print('get_reward', self.record)
        # print(self.avg_reward)
        # 如果avg_reward是nan
        if np.isnan(self.avg_reward):
            exit()
        return self.avg_reward
    
    def update_compactness(self, compactness, term1, term2):
        self.compactness = compactness
        self.term1 = term1
        self.term2 = term2

    def __dict__(self):
        return {
            'iteration': self.iteration,
            "strs": self.strs,
            "text": self.get_integrated_strs(),
            "shuffle_indexs": self.shuffle_indexs,
            "avg_reward": self.avg_reward,
            "compactness": self.compactness,
            "record": self.record,
            "all_record": self.all_record,
        }

    @classmethod
    def to_json(self, file_path, reflection_set_list):
        with open(file_path, 'a') as f:
            # add line
            for reflection_set in reflection_set_list:
                f.write(json.dumps(reflection_set.__dict__()) + '\n')

def get_all_eval_results(handlers, model_name, max_tokens, temperature, eval_str, target_trait=None):
    """
    Use the given model and parameters to obtain evaluation results through a series of handlers.  
    Supports parallel processing (if the model is not loaded from a local path) and trait-specific evaluation.

    Args:
        handlers: A list of evaluation handlers.
        model_name: The name of the LLM to use.
        max_tokens: The maximum number of tokens to generate.
        temperature: The temperature parameter for text generation.
        eval_str: The input string for evaluation.
        target_trait: Optional; if provided, only this specific trait will be evaluated.

    Returns:
        A list containing the evaluation results from each handler.
    """
    if target_trait is None:
        if '/' in model_name:
            # local model, not using parallel
            tmp_res_list = []
            for handler in handlers:
                tmp_res = handler.get_eval_results(model_name=model_name,
                                                shot_str=eval_str, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature)
                tmp_res_list.append(tmp_res)
        else:
            tmp_res_list = []
            for handler in handlers:
                tmp_res = handler.get_eval_results(model_name=model_name,
                                                shot_str=eval_str, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature)
                tmp_res_list.append(tmp_res)
    else:
        if '/' in model_name:
            # local model, not using parallel
            tmp_res_list = []
            for handler in handlers:
                tmp_res = handler.get_target_results(target_trait=target_trait,
                                                model_name=model_name,
                                                shot_str=eval_str, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature)
                tmp_res_list.append(tmp_res)
        else:
            tmp_res_list = []
            for handler in handlers:
                tmp_res = handler.get_target_results(target_trait=target_trait,
                                                    model_name=model_name,
                                                    shot_str=eval_str, 
                                                    max_tokens=max_tokens, 
                                                    temperature=temperature)
                tmp_res_list.append(tmp_res)
    return tmp_res_list
    
def calculate_effectiveness(reflection_set_list, 
                                handlers, 
                                target_trait,
                                model_name="GPT-4o",
                                max_tokens=8192,
                                temperature=0.01,
                                avg_mode="min"
                                ):
    """
    Calculate the effectiveness of a group of ReflectionSet.
    Effectiveness is measured by evaluating and summarizing the results of different reflection orders in each ReflectionSet.
    """
    for reflection_set in tqdm(reflection_set_list, desc="Calculate effectiveness"):
        eval_str_items = reflection_set.get_query_strs()
        for idx, shot_str in eval_str_items:
            eval_str = Controller.get_multi_shot_instruction(shot_str)
            tmp_res_list = get_all_eval_results(handlers, model_name, max_tokens, temperature, eval_str, target_trait)
            eval_str_tmp_res = merge_results(tmp_res_list, avg_mode)
            reflection_set.add_eval_record(idx, eval_str, eval_str_tmp_res[target_trait], all_dimention={
                'text': eval_str,
                'eval_res': eval_str_tmp_res
            })
    return [reflection_set.get_reward() for reflection_set in reflection_set_list]
        
def opt_main(args, target_trait, evaluator, router):
    """
    Main function for performing reflection optimization.  
    Iteratively generates new reflections, evaluates their effectiveness, and retains the best ones.

    Args:
        args: An object containing all command-line arguments.
        target_trait: The trait currently being optimized.
        evaluator: An `Evaluator` object used to perform evaluations.
        router: A `ModelRouter` object used to route requests to different LLMs.
    """

    output_dir = args.output_dir
    retriever_save_path = os.path.join(output_dir, f"{target_trait}_all_context.txt")
    reflection_set_save_path = os.path.join(output_dir, f"{target_trait}_reflection_set.jsonl")
    if os.path.exists(retriever_save_path) and os.path.exists(reflection_set_save_path):
        return
    # build evaluator
    candidate_mapping = evaluator.trait_mapping

    # opt: params
    MAX_ITER = args.max_iteration
    init_reflection_num = args.init_reflection_num
    num_item_shuffle = args.num_item_shuffle
    use_cot = args.use_cot
    optimization_beam_size = args.optimization_beam_size
    num_k_shot = args.num_k_shot
    summarization_beam_size = args.summarization_beam_size
    top_k = args.top_k
    words_limit = args.words_limit
    temperature = args.temperature
    max_tokens = args.max_tokens
    eval_model_name =  args.eval_model_name
    threshold = args.threshold
    model_name = args.model_name
    avg_mode = args.avg_mode
    use_task_description = args.use_task_description

    print(f"Optimizing for {target_trait}")
    # 0.a initialize reflection
    candidate_set = candidate_mapping[target_trait]
    retriever = Retriever(
        model_name=PATH_TO_SIMCSE_MODEL,
        corpus_texts=candidate_set,
        use_gpu=False,
    )
    retriever.save(retriever_save_path)

    reflection_set_list = []
    # use slide window to get some initial reflection_set
    for start in range(0, len(candidate_set), init_reflection_num):
        if start+init_reflection_num < len(candidate_set):
            texts_list = candidate_set[start:start+init_reflection_num] 
        else:
            rest_num = len(candidate_set) - start
            texts_list = candidate_set[start:] + random.sample(candidate_set, k=init_reflection_num-rest_num)
        reflection_set = ReflectionSet(texts_list, num_item_shuffle)
        reflection_set_list.append(reflection_set)
    
    # 0.b initialize scores
    prev_reward_list = calculate_effectiveness(reflection_set_list, 
                                            handlers=evaluator.get_handlers(), 
                                            target_trait=target_trait,
                                            model_name=model_name,
                                            max_tokens=max_tokens,
                                            temperature=0.01,
                                            avg_mode=avg_mode
                                            )
    ReflectionSet.to_json(reflection_set_save_path, reflection_set_list) # init reflection
    prev_reflections = reflection_set_list

    best_reflection, best_reward = None, -1

    for iter in tqdm(range(1, MAX_ITER), desc="Global Iteration"):
        print(f"Gloabl Iteration {iter}")
        # 1. Generate and filter new reflections
        new_reflections = []
        # 1.a Optimize relfection based on scores using LLM
        # random select optimization_beam_size query
        optimize_idxs = [random.sample(range(len(prev_reflections)), k=num_k_shot) for _ in range(optimization_beam_size)]
        optimize_input_strs_list = []
        for idxs in optimize_idxs:
            reflection_sets = [prev_reflections[idx] for idx in idxs]
            input_strs_score = [(reflection_set.get_raw_strs(), reflection_set.get_reward()) for reflection_set in reflection_sets]
            optimize_input_strs_list.extend(input_strs_score)
        
        task_description = None
        if use_task_description:
            task_description = get_task_description(target_trait, args.evaluation_system)

        if use_cot:            
            optimize_res = cot_optimize_reflections(router=router,
                                            shot_strs_list=optimize_input_strs_list, 
                                            model=model_name, 
                                            words_limit=words_limit, 
                                            task_description=task_description,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
        else:
            optimize_res = optimize_reflections(router=router,
                                            shot_strs_list=optimize_input_strs_list, 
                                            model=model_name, 
                                            words_limit=words_limit, 
                                            task_description=task_description,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
        new_reflections.extend([ReflectionSet(res, num_item_shuffle, iteration=iter) for res in optimize_res if len(res) > 0])
        # 1.b Summarize existing Contexts E
        # 总结现有的 contexts (所有已知的 reflections)
        all_strs = retriever.corpus_texts
        optimize_input_strs_list = []
        for _ in range(summarization_beam_size):
            random.shuffle(all_strs)
            optimize_input_strs_list.append(all_strs.copy())
        summarize_res = summary_reflections(router=router,
                                            shot_strs_list=optimize_input_strs_list, 
                                            model=model_name, 
                                            words_limit=words_limit, 
                                            task_description=None,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
        new_reflections.extend([ReflectionSet(res, num_item_shuffle, iteration=iter) for res in summarize_res if len(res) > 0])
        # 1.c filter new reflections based on eq 3 (variant: eq 14)
        tmp_all_reflection = ReflectionSet(all_strs, num_item_shuffle)
        new_reflection_scores = calculate_compactness(
            new_reflections, 
            prev_reflections, 
            tmp_all_reflection, 
            score_model_name=eval_model_name,
            batch_size=10
        )
        # filter topk reflections
        new_reflections = [new_reflections[idx] for idx in np.argsort(new_reflection_scores)[-top_k:]]
        # 2. a. Update task evaluation scores, keep top k
        reward_list = calculate_effectiveness(new_reflections, 
                                            handlers=evaluator.get_handlers(),
                                            target_trait=target_trait,
                                            model_name=model_name,
                                            max_tokens=max_tokens,
                                            temperature=0.01,
                                            avg_mode=avg_mode
                                            )
        ReflectionSet.to_json(reflection_set_save_path, new_reflections)
        # 2. b. Update reward, prev_reward_list, retriever corpus_texts, best_reflection, best_reward
        for idx, reward in enumerate(reward_list):
            if reward > best_reward:
                best_reward = reward
                best_reflection = new_reflections[idx]
                
            prev_reward_list.append(reward)
            prev_reflections.append(new_reflections[idx])
            retriever.dedup_and_add_corpus(new_reflections[idx].get_raw_strs(), threshold=threshold)
        retriever.save(retriever_save_path.replace(".txt", f"_{iter}.txt"))
        print(f"iter {iter}, best_reward: {best_reward}")
        print(f"best_reflection: {best_reflection.get_integrated_strs()}")
        
def get_task_description(target_trait, system):
    """
    Generate task description based on target trait and specific system.
    """
    if 'value' in system or 'personality' in system or 'moral' in system:
        return f"""The task is to perform a series of downstream tasks—
such as surveys, role-playing, brainstorming, or creative writing—while adopting and
demonstrating certain {system}-based traits. The traits are {target_trait}."""
    else:
        return f"""The task is to perform well in a series of knowledge-based and reasoning tasks, the core subject of which is {target_trait}."""

if __name__ == "__main__":

    arguments = argparse.ArgumentParser()
    arguments.add_argument("--max_iteration", type=int, default=5, help="max iteration")
    arguments.add_argument("--init_reflection_num", type=int, default=50, help="initial number of reflections per set")
    arguments.add_argument("--num_item_shuffle", type=int, default=0, help="number of shuffle")
    arguments.add_argument("--use_cot", type=bool, default=True, help="use cot or not")
    arguments.add_argument("--optimization_beam_size", type=int, default=3, help="optimization beam size")
    arguments.add_argument("--num_k_shot", type=int, default=1, help="number of k shot")
    arguments.add_argument("--summarization_beam_size", type=int, default=2, help="summarization beam size")
    arguments.add_argument("--avg_mode", type=str, default="min", choices=["min", "max", 'algorithmic_mean', 'geometric_mean', 'harmonic_mean'])
    arguments.add_argument("--top_k", type=int, default=3, help="top k reflections to keep")
    arguments.add_argument("--words_limit", type=int, default=200, help="words limit in optimization")
    arguments.add_argument("--temperature", type=float, default=1.0, help="temperature in optimization")
    arguments.add_argument("--max_tokens", type=int, default=1024, help="max tokens in optimization")
    arguments.add_argument("--eval_model_name", type=str, default="GPT-4o", help="evaluation model name")
    arguments.add_argument("--threshold", type=float, default=0.8, help="threshold for deduplication")
    arguments.add_argument("--output_dir", type=str, default="output/all_survey")
    arguments.add_argument("--model_name", type=str, default="GPT-4o")
    # vllm params
    arguments.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    arguments.add_argument("--pipeline_parallel_size", type=int, default=1)
    arguments.add_argument("--tensor_parallel_size", type=int, default=1)
    # evaluation params
    arguments.add_argument("--evaluation_system", type=str, default="value", choices=["value", "personality", "moral"])
    arguments.add_argument("--tasks", type=str, default="survey")
    arguments.add_argument("--use_task_description", type=bool, default=True)
    arguments.add_argument("--specific_traits", type=str, default="")
    args = arguments.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    router = ModelRouter(
        model_names=[args.model_name, args.eval_model_name],
        model_dir=None,
        temperature=args.temperature,
        top_p=0.95,
        max_model_len=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )

    evaluator = Evaluator(
        router=router,
            evaluation_system=args.evaluation_system,
            tasks=args.tasks.split(","),
        )
    trait_mapping = evaluator.get_trait_mapping()

    if args.specific_traits:
        specific_traits = args.specific_traits.replace(' ', '').split(",")
        for trait in specific_traits:
            assert trait in trait_mapping.keys(), f"{trait} is not in trait_mapping"
        target_traits = specific_traits
    else:
        target_traits = trait_mapping.keys()

    for target_trait in target_traits:
        # Prepare input data
        eval_save_dir = os.path.join(args.output_dir, f'{target_trait}_eval')
        evaluator = Evaluator(
            router=router,
            evaluation_system=args.evaluation_system,
            tasks=args.tasks.split(","),
            save_dir=eval_save_dir
        )
        opt_main(args, target_trait, evaluator, router)
 