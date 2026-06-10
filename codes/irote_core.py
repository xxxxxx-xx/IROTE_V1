"""
IROTE Core Algorithm - Paper-Aligned Implementation
===================================================
Fixes based on detailed paper analysis:
1. Evocativeness: Added logprob support + self-normalized importance weights
2. Compactness: Restored ProbabilityEstimator (prompt-based P(t1|t2))
3. SimCSE: Re-enabled for initialization and deduplication
4. Self-evaluation: Separate target_model and judge_model
5. Open tasks: Complete templates for all traits
6. Scoring: Fixed questionnaire scoring, increased M2 for ranking
"""

import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Reflection:
    text: str
    iteration: int = 0
    source: str = ""
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class TaskResponse:
    task_prompt: str
    reflection: str
    response: str
    iteration: int
    logprob: float = 0.0  # log p_e(y|x)
    q_score: float = 0.0  # q_ω(v|y,x) in [0,1]
    log_q: float = 0.0    # log q_ω(v|y,x)


# ============================================================
# Trait Descriptions (Complete for all systems)
# ============================================================

TRAIT_DESCRIPTIONS = {
    # BigFive
    "extraversion": "Extraversion reflects being energetic, talkative, assertive, socially engaged, enthusiastic, and comfortable initiating interaction.",
    "agreeableness": "Agreeableness reflects being compassionate, cooperative, trusting, helpful, forgiving, and considerate towards others.",
    "conscientiousness": "Conscientiousness reflects being organized, disciplined, reliable, thorough, ambitious, and goal-directed.",
    "neuroticism": "Neuroticism reflects tendencies towards anxiety, emotional instability, worry, moodiness, and vulnerability to stress.",
    "openness": "Openness reflects being curious, creative, imaginative, open to new experiences, appreciative of art, and intellectually flexible.",
    # STBHV
    "self-direction": "Self-direction reflects valuing independent thought and action—choosing, creating, exploring.",
    "stimulation": "Stimulation reflects valuing excitement, novelty and challenge in life.",
    "hedonism": "Hedonism reflects valuing pleasure or sensuous gratification for oneself.",
    "achievement": "Achievement reflects valuing personal success through demonstrating competence according to social standards.",
    "power": "Power reflects valuing social status and prestige, control or dominance over people and resources.",
    "security": "Security reflects valuing safety, harmony, and stability of society, relationships, and of self.",
    "conformity": "Conformity reflects valuing restraint of actions, inclinations, and impulses likely to upset or harm others.",
    "tradition": "Tradition reflects valuing respect, commitment, and acceptance of the customs and ideas that one's culture or religion provides.",
    "benevolence": "Benevolence reflects preserving and enhancing the welfare of those with whom one is in frequent personal contact.",
    "universalism": "Universalism reflects understanding, appreciation, tolerance, and protection for the welfare of all people and for nature.",
    # MFT
    "care": "Care/Harm reflects cherishing and protecting others, and empathizing with those who suffer.",
    "fairness": "Fairness/Cheating reflects rendering justice according to shared rules, and avoiding cheating.",
    "loyalty": "Loyalty/Betrayal reflects standing with your group, family, nation, or tribe.",
    "authority": "Authority/Subversion reflects obeying tradition and legitimate authority, and respecting those in power.",
    "sanctity": "Sanctity/Degradation reflects avoiding disgusting things, which are seen as unworthy of respect and protection.",
}


# ============================================================
# Prompt Templates
# ============================================================

def build_behavior_sampling_prompt(candidate_reflection: str, trait_name: str, trait_description: str, M1: int = 3) -> str:
    """E-Step for Compactness (Algorithm 1, line 3)."""
    return f"""You are inferring concrete behaviors from a self-reflection.

Target trait: {trait_name} - {trait_description}

Self-reflection:
"{candidate_reflection}"

If this reflection describes me, what concrete behaviors would I naturally exhibit?

Please generate {M1} specific, task-independent behaviors that:
1. Directly follow from the reflection above
2. Reveal the underlying trait implicitly through action, not labels
3. Are diverse from each other
4. Each be one concrete sentence (under 25 words)

Output a JSON list of strings only:
["behavior 1", "behavior 2", ...]"""


def build_compact_prompt(current_reflection: str, candidate_reflections: List[str], behavior_map: Dict[str, List[str]], trait_name: str, trait_description: str, max_words: int = 50) -> str:
    """M-Step for Compactness (Algorithm 1, line 5 / Eq.3)."""
    pairs_text = ""
    for i, ref in enumerate(candidate_reflections):
        behaviors = behavior_map.get(ref, [])
        beh_str = "; ".join(behaviors) if behaviors else "no behaviors sampled"
        pairs_text += f"\nReflection {i+1}: {ref}\n  Behaviors: {beh_str}\n"

    return f"""You are synthesizing a compact self-reflection from multiple candidates and their behaviors.

Target trait: {trait_name} - {trait_description}

Current reflection and candidates with their implied behaviors:
{pairs_text}

Your task: Create a single compact self-reflection that:
1. Preserves the COMMON behavior-driving patterns shared across all candidates
2. Can recover both the original reflections AND their implied behaviors
3. Removes irrelevant background details (age, family, location, religion)
4. Removes redundant expressions and stop words
5. Uses first-person format: "I [verb]..., e.g.: [concrete example]"
6. Stays under {max_words} words

Output ONLY the compact reflection text, nothing else:"""


def build_response_sampling_prompt(reflection: str, task_prompt: str) -> str:
    """E-Step for Evocativeness (Algorithm 1, line 7)."""
    return f"""Given the following insights about me:
{reflection}
Please make the following responses strictly align with these insights.

{task_prompt}"""


def build_revision_prompt(compacted_reflection: str, top_responses: List[Dict], bottom_responses: List[Dict], trait_name: str, trait_description: str, K: int = 10, max_words: int = 50) -> str:
    """M-Step for Evocativeness (Algorithm 1, line 11 / Eq.5)."""
    top_text = ""
    for i, r in enumerate(top_responses[:3]):
        top_text += f"\n[Good response {i+1}] (score: {r['score']:.2f})\nTask: {r['task'][:100]}\nResponse excerpt: {r['response'][:200]}\n"

    bottom_text = ""
    for i, r in enumerate(bottom_responses[:3]):
        bottom_text += f"\n[Poor response {i+1}] (score: {r['score']:.2f})\nTask: {r['task'][:100]}\nResponse excerpt: {r['response'][:200]}\n"

    return f"""You are optimizing a self-reflection to better elicit a target trait.

Target trait: {trait_name} - {trait_description}

Current reflection:
"{compacted_reflection}"

Responses that effectively reflect the trait:
{top_text}

Responses that fail to reflect the trait:
{bottom_text}

Analyze what patterns in the reflection led to good vs poor trait expression.
Then generate {K} revised self-reflections that:
1. Strengthen patterns from good responses
2. Avoid patterns from poor responses
3. Are generalizable across tasks (not overfitted)
4. Each under {max_words} words
5. Use first-person: "I [verb]..., e.g.: [example]"
6. Do NOT say "I am {trait_name}" or use explicit labels
7. Do NOT include demographics (age, gender, country, family)

Output a JSON list of strings:
["reflection 1", "reflection 2", ...]"""


def build_judge_prompt(trait_name: str, trait_description: str, task_prompt: str, response: str) -> str:
    """Judge prompt for q_ω(v|y,x)."""
    return f"""You are a psychological trait evaluator. Assess how strongly a response reflects a target trait.

Target trait: {trait_name}
Description: {trait_description}

Task given to the model:
{task_prompt}

Model's response:
{response}

Evaluate how strongly this response reflects the target trait.

Scoring guide:
- Score 5: Strongly and consistently reflects the trait through concrete behaviors, motivations, or choices
- Score 4: Moderately reflects the trait with some good behavioral indicators
- Score 3: Neutral or ambiguous - neither clearly reflects nor contradicts the trait
- Score 2: Weakly reflects or partially contradicts the trait
- Score 1: Strongly contradicts or completely lacks the trait

IMPORTANT:
- Do NOT reward explicit self-labeling (e.g., "I am extroverted")
- Reward concrete behaviors, motivations, emotional orientation, and decision patterns
- Consider tone, energy, and implicit indicators

Output JSON only:
{{"score": <1-5>, "confidence": <0.0-1.0>, "evidence": "brief explanation"}}"""


# ============================================================
# Compactness: ProbabilityEstimator (Prompt-based P(t1|t2))
# ============================================================

def get_eval_prompt(texta: str, textb: str, inverse: bool = False) -> str:
    """Estimate P(Text1|Text2) via prompting (from official code)."""
    pos_a, pos_b = ("1", "2") if not inverse else ("2", "1")
    if inverse:
        texta, textb = textb, texta
    return f"""In the context of language modeling, we want to estimate the conditional probability P(Text 1 | Text 2). Please provide a score from 0 to 10 to represent this probability, where 0 means P(Text 1 | Text 2) is essentially zero, and 10 means P(Text 1 | Text 2) is very close to one.

[Text {pos_a}]:
{texta}

[Text {pos_b}]:
{textb}

Score (representing P(Text 1 | Text 2)): """


def get_eval_prompt_entailment(texta: str, textb: str, inverse: bool = False) -> str:
    """Estimate P(Text1|Text2) via textual entailment."""
    pos_a, pos_b = ("1", "2") if not inverse else ("2", "1")
    if inverse:
        texta, textb = textb, texta
    return f"""On a scale from 0 to 10, where 0 means Text 1 provides absolutely no evidence for Text 2, and 10 means Text 1 completely and undeniably entails Text 2, how strongly does Text 1 support or imply Text 2?

[Text {pos_a}]:
{texta}

[Text {pos_b}]:
{textb}
Score: """


def get_eval_prompt_relatedness(texta: str, textb: str, inverse: bool = False) -> str:
    """Estimate P(Text1|Text2) via relatedness."""
    pos_a, pos_b = ("1", "2") if not inverse else ("2", "1")
    if inverse:
        texta, textb = textb, texta
    return f"""On a scale from 0 to 10, where 0 means Text 1 is completely unrelated to Text 2, and 10 means Text 1 is almost identical to Text 2, how related are Text 1 and Text 2?

[Text {pos_a}]:
{texta}

[Text {pos_b}]:
{textb}
Score: """


def extract_score(response: str) -> float:
    """Extract numerical score from response."""
    try:
        return float(response.strip())
    except:
        digits = re.findall(r"\d+", response)
        if digits:
            return float(digits[0])
        return None


class ProbabilityEstimator:
    """Estimate P(t1|t2) via prompting (closed-source friendly)."""

    def __init__(self, prompt_types: List[int] = None):
        if prompt_types is None:
            prompt_types = [0, 1, 2]  # all three prompts
        all_funcs = [get_eval_prompt, get_eval_prompt_entailment, get_eval_prompt_relatedness]
        self.eval_funcs = [all_funcs[i] for i in prompt_types if i < len(all_funcs)]

    def get_score(self, texta: str, textb: str, router, model_name: str) -> float:
        """Estimate P(texta|textb) by asking LLM to score."""
        messages = []
        for is_inverse in [False, True]:
            for eval_func in self.eval_funcs:
                prompt = eval_func(texta, textb, is_inverse)
                messages.append([{"role": "user", "content": prompt}])

        responses = router.request_llm(
            conversations=messages,
            model=model_name,
            max_length=100,
            temperature=0.001,
        )

        scores = []
        for resp in responses:
            s = extract_score(resp)
            if s is not None:
                scores.append(s)

        return np.mean(scores) * 0.1 if scores else 0.5


def compute_compactness_pmi(
    router,
    target_reflection: str,
    candidate_reflections: List[str],
    all_reflections_text: str,
    model_name: str,
) -> Tuple[float, float, float]:
    """
    Compute compactness using PMI (Eq.2/3).
    Returns: (compactness_score, term1, term2)

    term1 = sum_k P(e|e_k) * [log P(e_k) + log P(s_k)]  (recovery)
    term2 = log P(E|e)  (redundancy)
    """
    estimator = ProbabilityEstimator()
    eps = 1e-6

    # Term 1: Recovery - can ê recover each e_k?
    term1_scores = []
    for cand in candidate_reflections:
        p_e_given_ek = estimator.get_score(target_reflection, cand, router, model_name)
        p_ek_given_e = estimator.get_score(cand, target_reflection, router, model_name)
        # PMI(e, e_k) = log P(e_k|e) + log P(e) - log P(e_k) ≈ log P(e_k|e)
        term1_scores.append(p_e_given_ek * np.log(p_ek_given_e + eps))

    term1 = np.mean(term1_scores) if term1_scores else 0.0

    # Term 2: Redundancy - is ê too similar to E?
    p_E_given_e = estimator.get_score(all_reflections_text, target_reflection, router, model_name)
    term2 = np.log(p_E_given_e + eps)

    # Compactness = recovery - redundancy
    compactness = term1 - term2

    return compactness, term1, term2


# ============================================================
# Parsing Utilities
# ============================================================

def extract_json_list(text: str) -> List[str]:
    """Extract a JSON list from LLM response text."""
    if not text or not text.strip():
        return []

    text = text.strip()
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Try JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(item).strip() for item in result if item and str(item).strip()]
        except json.JSONDecodeError:
            pass

    # Try full JSON
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(item).strip() for item in result if item and str(item).strip()]
        if isinstance(result, dict):
            for key in ["candidates", "reflections", "behaviors", "items"]:
                if key in result and isinstance(result[key], list):
                    return [
                        str(item.get("reflection", item) if isinstance(item, dict) else item).strip()
                        for item in result[key] if item
                    ]
    except json.JSONDecodeError:
        pass

    # Fallback: numbered items
    items = re.findall(r'\d+\.\s*["\']?(.+?)["\']?\s*(?:\n|$)', text)
    if items and len(items) >= 2:
        return [item.strip().strip('"').strip("'") for item in items if item.strip()]

    # Fallback: quoted strings
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted and len(quoted) >= 2:
        return [q.strip() for q in quoted if len(q.strip()) > 10]

    return []


def extract_judge_score(text: str) -> Tuple[float, float, str]:
    """Extract score, confidence, and evidence from judge response."""
    if not text or not text.strip():
        return 0.5, 0.0, "empty_response"

    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = data.get("score", None)
            confidence = data.get("confidence", 0.5)
            evidence = data.get("evidence", "")
            if score is not None:
                score = float(score)
                normalized = (score - 1) / 4.0  # map [1,5] to [0,1]
                return normalized, float(confidence), str(evidence)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    numbers = re.findall(r'\b([1-5])\b', text)
    if numbers:
        return (float(numbers[0]) - 1) / 4.0, 0.3, "fallback_parse"
    return 0.5, 0.0, "parse_failed"


def extract_questionnaire_score(response: str, scale: int = 5, reverse: bool = False) -> Optional[float]:
    """Extract numerical score from questionnaire response."""
    numbers = re.findall(r'\b(\d+)\b', response.strip())
    if numbers:
        raw = int(numbers[0])
        if 1 <= raw <= scale:
            if reverse:
                raw = scale + 1 - raw
            return (raw - 1) / (scale - 1)
    return None


# ============================================================
# Core Algorithm Functions
# ============================================================

def sample_behaviors(
    router,
    candidate_reflections: List[str],
    trait_name: str,
    trait_description: str,
    model_name: str,
    M1: int = 3,
    temperature: float = 0.9,
    max_tokens: int = 512,
) -> Dict[str, List[str]]:
    """Compactness E-Step (Algorithm 1, lines 2-4)."""
    behavior_map = {}
    for cand in candidate_reflections:
        prompt = build_behavior_sampling_prompt(
            candidate_reflection=cand,
            trait_name=trait_name,
            trait_description=trait_description,
            M1=M1,
        )
        messages = [[{"role": "user", "content": prompt}]]
        responses = router.request_llm(
            conversations=messages,
            model=model_name,
            max_length=max_tokens,
            temperature=temperature,
        )
        behaviors = extract_json_list(responses[0]) if responses else []
        behavior_map[cand] = behaviors[:M1]
    return behavior_map


def compact_reflection(
    router,
    current_reflection: str,
    candidate_reflections: List[str],
    behavior_map: Dict[str, List[str]],
    trait_name: str,
    trait_description: str,
    model_name: str,
    max_words: int = 50,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """Compactness M-Step (Algorithm 1, line 5 / Eq.3)."""
    prompt = build_compact_prompt(
        current_reflection=current_reflection,
        candidate_reflections=candidate_reflections,
        behavior_map=behavior_map,
        trait_name=trait_name,
        trait_description=trait_description,
        max_words=max_words,
    )
    messages = [[{"role": "user", "content": prompt}]]
    responses = router.request_llm(
        conversations=messages,
        model=model_name,
        max_length=max_tokens,
        temperature=temperature,
    )
    result = responses[0].strip() if responses else current_reflection
    result = re.sub(r'^["\']|["\']$', '', result)
    return result


def sample_and_evaluate_responses(
    router,
    reflection: str,
    task_prompts: List[str],
    trait_name: str,
    trait_description: str,
    target_model: str,
    judge_model: str = None,
    M2: int = 6,
    response_temp: float = 0.7,
    judge_temp: float = 0.0,
    max_tokens: int = 1024,
) -> Tuple[List[List[str]], List[List[float]], List[List[float]], List[List[float]]]:
    """
    Evocativeness E-Step (Algorithm 1, lines 6-9).
    Returns: (response_sets, q_scores, log_probs, confidences)
    """
    if judge_model is None:
        judge_model = target_model

    all_responses = []
    all_q_scores = []
    all_log_probs = []
    all_confidences = []

    for task in task_prompts:
        prompt = build_response_sampling_prompt(reflection=reflection, task_prompt=task)
        task_responses = []
        task_q_scores = []
        task_log_probs = []
        task_confidences = []

        for _ in range(M2):
            # Sample response from target model
            messages = [[{"role": "user", "content": prompt}]]
            responses = router.request_llm(
                conversations=messages,
                model=target_model,
                max_length=max_tokens,
                temperature=response_temp,
            )
            response_text = responses[0] if responses else ""
            task_responses.append(response_text)

            # Logprob: use uniform weight (closed-source without logprobs)
            # TODO: If API supports logprobs, compute seq_logprob here
            log_prob = 0.0
            task_log_probs.append(log_prob)

            # Evaluate with q_ω (judge model, separate from target)
            judge_prompt = build_judge_prompt(
                trait_name=trait_name,
                trait_description=trait_description,
                task_prompt=task,
                response=response_text,
            )
            judge_messages = [[{"role": "user", "content": judge_prompt}]]
            judge_responses = router.request_llm(
                conversations=judge_messages,
                model=judge_model,
                max_length=256,
                temperature=judge_temp,
            )
            q_normalized, confidence, _ = extract_judge_score(judge_responses[0] if judge_responses else "")
            task_q_scores.append(q_normalized)
            task_confidences.append(confidence)

        all_responses.append(task_responses)
        all_q_scores.append(task_q_scores)
        all_log_probs.append(task_log_probs)
        all_confidences.append(task_confidences)

    return all_responses, all_q_scores, all_log_probs, all_confidences


def compute_R2(
    q_scores: List[List[float]],
    log_probs: List[List[float]] = None,
    confidences: List[List[float]] = None,
    use_confidence_weighting: bool = True,
) -> float:
    """
    Compute R2(e) - Evocativeness score (Eq.5).
    R2 = (1/N) * sum_i sum_j p_e(y|x) * log q_ω(v|y,x)

    With confidence weighting: weight = confidence * (1/M2) instead of uniform 1/M2
    """
    eps = 1e-6
    total = 0.0
    count = 0

    for i, task_scores in enumerate(q_scores):
        for j, q in enumerate(task_scores):
            # Weight: p_e(y|x) if available, else confidence-weighted uniform
            if log_probs is not None and i < len(log_probs) and j < len(log_probs[i]) and log_probs[i][j] != 0.0:
                weight = np.exp(log_probs[i][j])
            elif use_confidence_weighting and confidences is not None and i < len(confidences) and j < len(confidences[i]):
                # Self-normalized importance weight using judge confidence
                conf = confidences[i][j]
                weight = conf / max(sum(confidences[i]), eps)
            else:
                weight = 1.0 / max(len(task_scores), 1)

            log_q = np.log(q + eps)
            total += weight * log_q
            count += 1

    return total / max(count, 1)


def generate_revised_candidates(
    router,
    compacted_reflection: str,
    task_prompts: List[str],
    response_sets: List[List[str]],
    q_scores: List[List[float]],
    trait_name: str,
    trait_description: str,
    model_name: str,
    K: int = 10,
    max_words: int = 50,
    temperature: float = 0.6,
    max_tokens: int = 1024,
) -> List[str]:
    """Evocativeness M-Step (Algorithm 1, line 11 / Eq.5)."""
    scored_responses = []
    for i, (task, responses, scores) in enumerate(zip(task_prompts, response_sets, q_scores)):
        for resp, score in zip(responses, scores):
            scored_responses.append({"task": task, "response": resp, "score": score})

    scored_responses.sort(key=lambda x: x["score"], reverse=True)
    n = max(len(scored_responses) // 3, 1)
    top_responses = scored_responses[:n]
    bottom_responses = scored_responses[-n:]

    prompt = build_revision_prompt(
        compacted_reflection=compacted_reflection,
        top_responses=top_responses,
        bottom_responses=bottom_responses,
        trait_name=trait_name,
        trait_description=trait_description,
        K=K,
        max_words=max_words,
    )
    messages = [[{"role": "user", "content": prompt}]]
    responses = router.request_llm(
        conversations=messages,
        model=model_name,
        max_length=max_tokens,
        temperature=temperature,
    )
    candidates = extract_json_list(responses[0]) if responses else []
    return candidates[:K]


def rank_candidates(
    router,
    candidates: List[str],
    validation_tasks: List[str],
    trait_name: str,
    trait_description: str,
    target_model: str,
    judge_model: str = None,
    K: int = 3,
    M2: int = 4,
    beta: float = 1.0,
    max_words: int = 50,
) -> List[Reflection]:
    """
    Candidate ranking (Algorithm 1, lines 12-13).
    Uses PMI-based compactness + evocativeness with confidence weighting.
    """
    if judge_model is None:
        judge_model = target_model

    ranked = []
    for cand_text in candidates:
        # Sample validation responses
        resp_sets, q_scores, _, confidences = sample_and_evaluate_responses(
            router=router,
            reflection=cand_text,
            task_prompts=validation_tasks[:3],
            trait_name=trait_name,
            trait_description=trait_description,
            target_model=target_model,
            judge_model=judge_model,
            M2=M2,
        )

        # Evocativeness with confidence weighting
        r2 = compute_R2(q_scores, confidences=confidences, use_confidence_weighting=True)
        all_q = [q for task_q in q_scores for q in task_q]
        evocativeness = np.mean(all_q) if all_q else 0.0

        # Compactness: PMI-based (using ProbabilityEstimator)
        all_cand_text = "\n".join(candidates)
        compactness_pmi, term1, term2 = compute_compactness_pmi(
            router=router,
            target_reflection=cand_text,
            candidate_reflections=candidates,
            all_reflections_text=all_cand_text,
            model_name=judge_model,
        )

        # Normalize compactness to [0, 1]
        compactness = max(0, min(1, (compactness_pmi + 5) / 10))

        # Joint objective (Eq.1): compactness + β * evocativeness
        final_score = compactness + beta * evocativeness

        # Length penalty
        words = cand_text.split()
        length_penalty = max(0, (len(words) - max_words) / max_words)

        ref = Reflection(
            text=cand_text,
            source="revise",
            scores={
                "final_score": final_score,
                "compactness": compactness,
                "compactness_pmi": compactness_pmi,
                "evocativeness": evocativeness,
                "r2": r2,
                "term1": term1,
                "term2": term2,
                "length_penalty": length_penalty,
                "length": len(words),
            },
        )
        ranked.append(ref)

    ranked.sort(key=lambda x: x.scores["final_score"], reverse=True)
    return ranked


# ============================================================
# Full IROTE Iteration
# ============================================================

def run_irote_iteration(
    router,
    iteration: int,
    current_reflection: str,
    candidate_reflections: List[str],
    optimize_tasks: List[str],
    trait_name: str,
    trait_description: str,
    target_model: str,
    judge_model: str = None,
    K: int = 10,
    M1: int = 3,
    M2: int = 6,
    beta: float = 1.0,
    max_words: int = 50,
) -> Dict:
    """Run one full IROTE iteration (Algorithm 1)."""
    if judge_model is None:
        judge_model = target_model

    print(f"\n{'='*60}")
    print(f"IROTE Iteration {iteration}")
    print(f"{'='*60}")

    # Step 1-4: Compactness E-Step
    print(f"\n[Step 1] Compactness E-Step: Sampling {M1} behaviors per candidate...")
    behavior_map = sample_behaviors(
        router=router,
        candidate_reflections=candidate_reflections,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=target_model,
        M1=M1,
    )
    total_behaviors = sum(len(v) for v in behavior_map.values())
    print(f"  Generated {total_behaviors} behaviors for {len(behavior_map)} candidates")

    # Step 5: Compactness M-Step
    print(f"\n[Step 2] Compactness M-Step: Synthesizing compact reflection...")
    compacted = compact_reflection(
        router=router,
        current_reflection=current_reflection,
        candidate_reflections=candidate_reflections,
        behavior_map=behavior_map,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=target_model,
        max_words=max_words,
    )
    print(f"  Compacted ({len(compacted.split())} words): {compacted[:100]}...")

    # Step 6-9: Evocativeness E-Step
    print(f"\n[Step 3] Evocativeness E-Step: Sampling {M2} responses per task...")
    response_sets, q_scores, log_probs, confidences = sample_and_evaluate_responses(
        router=router,
        reflection=compacted,
        task_prompts=optimize_tasks,
        trait_name=trait_name,
        trait_description=trait_description,
        target_model=target_model,
        judge_model=judge_model,
        M2=M2,
    )

    # Compute R2 with confidence weighting
    r2 = compute_R2(q_scores, log_probs, confidences, use_confidence_weighting=True)
    all_q = [q for task_q in q_scores for q in task_q]
    avg_q = np.mean(all_q) if all_q else 0.0
    print(f"  R2 score: {r2:.4f}, Avg q_ω: {avg_q:.4f}")

    # Step 11: Evocativeness M-Step
    print(f"\n[Step 4] Evocativeness M-Step: Generating {K} revised candidates...")
    revised_candidates = generate_revised_candidates(
        router=router,
        compacted_reflection=compacted,
        task_prompts=optimize_tasks,
        response_sets=response_sets,
        q_scores=q_scores,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=target_model,
        K=K,
        max_words=max_words,
    )
    print(f"  Generated {len(revised_candidates)} candidates")

    # Step 12-13: Rank candidates
    print(f"\n[Step 5] Ranking candidates...")
    ranked = rank_candidates(
        router=router,
        candidates=revised_candidates,
        validation_tasks=optimize_tasks[:5],
        trait_name=trait_name,
        trait_description=trait_description,
        target_model=target_model,
        judge_model=judge_model,
        K=3,
        M2=4,
        beta=beta,
        max_words=max_words,
    )

    # Select best
    if ranked:
        best = ranked[0]
    else:
        # Fallback
        all_cand_text = "\n".join(candidate_reflections)
        compactness_pmi, _, _ = compute_compactness_pmi(
            router=router,
            target_reflection=compacted,
            candidate_reflections=candidate_reflections,
            all_reflections_text=all_cand_text,
            model_name=judge_model,
        )
        compactness = max(0, min(1, (compactness_pmi + 5) / 10))
        best = Reflection(
            text=compacted,
            iteration=iteration,
            source="compact_fallback",
            scores={
                "final_score": compactness + beta * avg_q,
                "compactness": compactness,
                "compactness_pmi": compactness_pmi,
                "evocativeness": avg_q,
                "r2": r2,
                "length_penalty": 0.0,
                "length": len(compacted.split()),
            },
        )

    print(f"\n  Best reflection (score={best.scores.get('final_score', 0):.4f}):")
    print(f"  {best.text}")

    return {
        "iteration": iteration,
        "current_reflection": current_reflection,
        "compacted_reflection": compacted,
        "behavior_map": {k: v for k, v in behavior_map.items()},
        "response_sets": response_sets,
        "q_scores": q_scores,
        "confidences": confidences,
        "r2_score": r2,
        "avg_q_score": avg_q,
        "revised_candidates": revised_candidates,
        "ranked_candidates": ranked,
        "best_reflection": best,
    }
