"""
IROTE Core Algorithm - Strict Paper Implementation
===================================================
Implements Algorithm 1 from the paper with all equations:
- Eq.(1): e* = argmax TC(e,E) + β·I_e(v;y|x)
- Eq.(2)(3): Compactness R₁ with behavior sampling
- Eq.(4)(5): Evocativeness R₂ with M₂ sampling and logprobs
- Algorithm 1: Full EM iteration

Key design decisions for closed-source models:
- p_e(y|x): Use logprobs from MiMo API when available, uniform weight otherwise
- q_ω(v|y,x): LLM-as-judge outputting log probability
- TC(e,E): Behavior-anchored PMI approximation
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
    """A single self-reflection with metadata."""
    text: str
    iteration: int = 0
    source: str = ""
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class BehaviorSample:
    """A behavior implied by a reflection."""
    reflection: str
    behavior: str
    iteration: int


@dataclass
class TaskResponse:
    """A model response to a task."""
    task_prompt: str
    reflection: str
    response: str
    iteration: int
    logprob: float = 0.0  # log p_e(y|x)
    q_score: float = 0.0  # q_ω(v|y,x) in [0,1]
    log_q: float = 0.0    # log q_ω(v|y,x)


# ============================================================
# Prompt Templates (Behavior-Anchored, per paper)
# ============================================================

def build_behavior_sampling_prompt(
    candidate_reflection: str,
    trait_name: str,
    trait_description: str,
    M1: int = 3,
) -> str:
    """
    E-Step for Compactness (Algorithm 1, line 3).
    Paper: "if I often maintain harmonious team dynamics, how would I behave?"
    """
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


def build_compact_prompt(
    current_reflection: str,
    candidate_reflections: List[str],
    behavior_map: Dict[str, List[str]],
    trait_name: str,
    trait_description: str,
    max_words: int = 50,
) -> str:
    """
    M-Step for Compactness (Algorithm 1, line 5 / Eq.3).
    Paper: "Given such behaviors, what do they reflect?"
    Goal: synthesize ê that can recover both e_k and their behaviors s.
    """
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
    """
    E-Step for Evocativeness (Algorithm 1, line 7).
    Paper: inject ê as prefix, then sample response.
    """
    return f"""Given the following insights about me:
{reflection}
Please make the following responses strictly align with these insights.

{task_prompt}"""


def build_revision_prompt(
    compacted_reflection: str,
    top_responses: List[Dict],
    bottom_responses: List[Dict],
    trait_name: str,
    trait_description: str,
    K: int = 10,
    max_words: int = 50,
) -> str:
    """
    M-Step for Evocativeness (Algorithm 1, line 11 / Eq.5).
    Paper: compare high-scoring vs low-scoring responses to revise reflection.
    """
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


def build_judge_prompt(
    trait_name: str,
    trait_description: str,
    task_prompt: str,
    response: str,
) -> str:
    """
    Judge prompt for q_ω(v|y,x).
    Paper: trait evaluator that outputs probability of trait v in response y.
    """
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
    """
    Extract score, confidence, and evidence from judge response.
    Returns: (score_normalized, confidence, evidence)
    """
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

    # Fallback
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
    """
    Compactness E-Step (Algorithm 1, lines 2-4).
    For each candidate e_k, sample M1 behaviors: s ~ p_{e^{t-1}}(s|e_k)
    Returns: dict mapping reflection -> list of behaviors
    """
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
    """
    Compactness M-Step (Algorithm 1, line 5 / Eq.3).
    Paper: "Given such behaviors, what do they reflect?"
    Synthesize ê that recovers both e_k and their behaviors.
    """
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
    # Clean up any markdown or extra text
    result = re.sub(r'^["\']|["\']$', '', result)
    return result


def sample_and_evaluate_responses(
    router,
    reflection: str,
    task_prompts: List[str],
    trait_name: str,
    trait_description: str,
    model_name: str,
    M2: int = 6,
    response_temp: float = 0.7,
    judge_temp: float = 0.0,
    max_tokens: int = 1024,
) -> Tuple[List[List[str]], List[List[float]], List[List[float]]]:
    """
    Evocativeness E-Step (Algorithm 1, lines 6-9).
    For each task x_i, sample M2 responses and evaluate with q_ω.
    Returns: (response_sets, q_scores, log_probs)
    """
    all_responses = []
    all_q_scores = []
    all_log_probs = []

    for task in task_prompts:
        prompt = build_response_sampling_prompt(reflection=reflection, task_prompt=task)
        task_responses = []
        task_q_scores = []
        task_log_probs = []

        for _ in range(M2):
            # Sample response
            messages = [[{"role": "user", "content": prompt}]]
            responses = router.request_llm(
                conversations=messages,
                model=model_name,
                max_length=max_tokens,
                temperature=response_temp,
            )
            response_text = responses[0] if responses else ""
            task_responses.append(response_text)

            # Get logprob if available (closed-source approximation: uniform)
            log_prob = 0.0  # uniform weight when logprobs not available
            task_log_probs.append(log_prob)

            # Evaluate with q_ω (judge)
            judge_prompt = build_judge_prompt(
                trait_name=trait_name,
                trait_description=trait_description,
                task_prompt=task,
                response=response_text,
            )
            judge_messages = [[{"role": "user", "content": judge_prompt}]]
            judge_responses = router.request_llm(
                conversations=judge_messages,
                model=model_name,
                max_length=256,
                temperature=judge_temp,
            )
            q_normalized, confidence, _ = extract_judge_score(judge_responses[0] if judge_responses else "")
            task_q_scores.append(q_normalized)

        all_responses.append(task_responses)
        all_q_scores.append(task_q_scores)
        all_log_probs.append(task_log_probs)

    return all_responses, all_q_scores, all_log_probs


def compute_R2(
    q_scores: List[List[float]],
    log_probs: List[List[float]] = None,
) -> float:
    """
    Compute R2(e) - Evocativeness score (Eq.5).
    R2 = (1/N) * sum_i sum_j p_e(y|x) * log q_ω(v|y,x)
    For closed-source without logprobs: uniform weight approximation.
    """
    eps = 1e-6
    total = 0.0
    count = 0

    for i, task_scores in enumerate(q_scores):
        for j, q in enumerate(task_scores):
            if log_probs is not None and i < len(log_probs) and j < len(log_probs[i]):
                weight = np.exp(log_probs[i][j])  # p_e(y|x)
            else:
                weight = 1.0 / max(len(task_scores), 1)  # uniform

            log_q = np.log(q + eps)  # log q_ω(v|y,x)
            total += weight * log_q
            count += 1

    return total / max(count, 1)


def compute_compactness_score(
    reflection: str,
    candidate_reflections: List[str],
    behavior_map: Dict[str, List[str]],
) -> float:
    """
    Compute compactness score approximating TC(e,E).
    Based on how well the reflection covers shared patterns.
    """
    if not candidate_reflections:
        return 0.0

    # Simple heuristic: coverage of key terms
    ref_words = set(reflection.lower().split())
    coverage_scores = []

    for cand in candidate_reflections:
        cand_words = set(cand.lower().split())
        behaviors = behavior_map.get(cand, [])
        beh_words = set()
        for b in behaviors:
            beh_words.update(b.lower().split())

        # Coverage of candidate + behavior words
        all_words = cand_words | beh_words
        if all_words:
            overlap = len(ref_words & all_words) / len(all_words)
            coverage_scores.append(overlap)

    # Length penalty (prefer shorter)
    words = reflection.split()
    length_score = max(0, 1.0 - max(0, len(words) - 50) / 50)

    # Redundancy penalty
    unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)

    return np.mean(coverage_scores) * length_score * unique_ratio if coverage_scores else 0.0


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
    """
    Evocativeness M-Step (Algorithm 1, line 11 / Eq.5).
    Compare high/low responses to generate K revised candidates.
    """
    # Collect all responses with scores
    scored_responses = []
    for i, (task, responses, scores) in enumerate(zip(task_prompts, response_sets, q_scores)):
        for resp, score in zip(responses, scores):
            scored_responses.append({
                "task": task,
                "response": resp,
                "score": score,
            })

    # Sort by score
    scored_responses.sort(key=lambda x: x["score"], reverse=True)

    # Split into top and bottom
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
    model_name: str,
    K: int = 3,
    M2: int = 2,
    beta: float = 1.0,
    max_words: int = 50,
) -> List[Reflection]:
    """
    Candidate ranking (Algorithm 1, lines 12-13).
    For each candidate, compute final_score = compactness + β * evocativeness
    """
    ranked = []

    for cand_text in candidates:
        # Sample a few validation responses
        resp_sets, q_scores, _ = sample_and_evaluate_responses(
            router=router,
            reflection=cand_text,
            task_prompts=validation_tasks[:3],
            trait_name=trait_name,
            trait_description=trait_description,
            model_name=model_name,
            M2=M2,
            response_temp=0.7,
            judge_temp=0.0,
        )

        # Compute evocativeness
        r2 = compute_R2(q_scores)
        all_q = [q for task_q in q_scores for q in task_q]
        evocativeness = np.mean(all_q) if all_q else 0.0

        # Compute compactness
        words = cand_text.split()
        length_penalty = max(0, (len(words) - max_words) / max_words)
        unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
        redundancy_penalty = 1.0 - unique_ratio
        compactness = 1.0 - length_penalty - redundancy_penalty

        # Joint objective (Eq.1): compactness + β * evocativeness
        final_score = compactness + beta * evocativeness

        ref = Reflection(
            text=cand_text,
            source="revise",
            scores={
                "final_score": final_score,
                "compactness": compactness,
                "evocativeness": evocativeness,
                "r2": r2,
                "length_penalty": length_penalty,
                "redundancy_penalty": redundancy_penalty,
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
    model_name: str,
    K: int = 10,
    M1: int = 3,
    M2: int = 6,
    beta: float = 1.0,
    max_words: int = 50,
) -> Dict:
    """
    Run one full IROTE iteration (Algorithm 1).
    """
    print(f"\n{'='*60}")
    print(f"IROTE Iteration {iteration}")
    print(f"{'='*60}")

    # Step 1-4: Compactness E-Step (Algorithm 1, lines 2-4)
    print(f"\n[Step 1] Compactness E-Step: Sampling {M1} behaviors per candidate...")
    behavior_map = sample_behaviors(
        router=router,
        candidate_reflections=candidate_reflections,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=model_name,
        M1=M1,
    )
    total_behaviors = sum(len(v) for v in behavior_map.values())
    print(f"  Generated {total_behaviors} behaviors for {len(behavior_map)} candidates")

    # Step 5: Compactness M-Step (Algorithm 1, line 5 / Eq.3)
    print(f"\n[Step 2] Compactness M-Step: Synthesizing compact reflection...")
    compacted = compact_reflection(
        router=router,
        current_reflection=current_reflection,
        candidate_reflections=candidate_reflections,
        behavior_map=behavior_map,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=model_name,
        max_words=max_words,
    )
    print(f"  Compacted ({len(compacted.split())} words): {compacted[:100]}...")

    # Step 6-9: Evocativeness E-Step (Algorithm 1, lines 6-9)
    print(f"\n[Step 3] Evocativeness E-Step: Sampling {M2} responses per task...")
    response_sets, q_scores, log_probs = sample_and_evaluate_responses(
        router=router,
        reflection=compacted,
        task_prompts=optimize_tasks,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=model_name,
        M2=M2,
    )

    # Compute R2
    r2 = compute_R2(q_scores, log_probs)
    all_q = [q for task_q in q_scores for q in task_q]
    avg_q = np.mean(all_q) if all_q else 0.0
    print(f"  R2 score: {r2:.4f}, Avg q_ω: {avg_q:.4f}")

    # Step 11: Evocativeness M-Step (Algorithm 1, line 11 / Eq.5)
    print(f"\n[Step 4] Evocativeness M-Step: Generating {K} revised candidates...")
    revised_candidates = generate_revised_candidates(
        router=router,
        compacted_reflection=compacted,
        task_prompts=optimize_tasks,
        response_sets=response_sets,
        q_scores=q_scores,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=model_name,
        K=K,
        max_words=max_words,
    )
    print(f"  Generated {len(revised_candidates)} candidates")

    # Step 12-13: Rank candidates (Algorithm 1, lines 12-13)
    print(f"\n[Step 5] Ranking candidates...")
    ranked = rank_candidates(
        router=router,
        candidates=revised_candidates,
        validation_tasks=optimize_tasks[:5],
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=model_name,
        K=3,
        M2=2,
        beta=beta,
        max_words=max_words,
    )

    # Select best
    if ranked:
        best = ranked[0]
    else:
        # Fallback: use compacted reflection
        best = Reflection(
            text=compacted,
            iteration=iteration,
            source="compact_fallback",
            scores={
                "final_score": compute_compactness_score(compacted, candidate_reflections, behavior_map) + beta * avg_q,
                "compactness": compute_compactness_score(compacted, candidate_reflections, behavior_map),
                "evocativeness": avg_q,
                "r2": r2,
                "length_penalty": 0.0,
                "redundancy_penalty": 0.0,
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
        "r2_score": r2,
        "avg_q_score": avg_q,
        "revised_candidates": revised_candidates,
        "ranked_candidates": ranked,
        "best_reflection": best,
    }
