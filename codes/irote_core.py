"""
IROTE Core Algorithm Implementation
====================================
Implements the missing algorithmic components from the paper:
- Compactness E-Step: Behavior sampling (Algorithm 1, line 3)
- Compactness M-Step: Reflection refinement via R1 (Eq.3)
- Evocativeness E-Step: Response sampling + q_omega scoring (Algorithm 1, lines 6-10)
- Evocativeness M-Step: Reflection revision via R2 (Eq.5)
- Candidate ranking and selection (Algorithm 1, lines 12-13)

Paper reference: "IROTE: Human-Like Traits Elicitation of Large Language Model
via In-Context Self-Reflective Optimization" (AAAI-26)
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
    """A single self-reflection text with metadata."""
    text: str
    iteration: int = 0
    source: str = ""  # "init", "compact", "revise"
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return {
            "text": self.text,
            "iteration": self.iteration,
            "source": self.source,
            "scores": self.scores,
        }


@dataclass
class BehaviorSample:
    """A behavior implied by a reflection."""
    reflection_text: str
    behavior: str
    iteration: int


@dataclass
class TaskResponse:
    """A model response to a task, conditioned on a reflection."""
    task_prompt: str
    reflection_text: str
    response: str
    iteration: int
    score: float = 0.0
    judge_reason: str = ""


# ============================================================
# Prompt Templates (from paper + appendix)
# ============================================================

def build_behavior_sampling_prompt(
    current_reflection: str,
    candidate_reflection: str,
    trait_name: str,
    trait_description: str,
    M1: int = 3,
) -> str:
    """
    Compactness E-Step prompt (Algorithm 1, line 3).
    Paper: "we instruct the LLM to produce behavior s that it considers
    connected the reflection ek, when conditioned on e_{t-1}
    (analogously, if I often maintain harmonious team dynamics,
    how would I behave?)"
    """
    return f"""You are inferring concrete behaviors from a self-reflection.

Target trait: {trait_name}
Trait description: {trait_description}

Current global reflection:
{current_reflection}

Candidate reflection to analyze:
{candidate_reflection}

Please generate {M1} concrete, task-independent behaviors that would naturally follow from the candidate reflection above. These behaviors should:
1. Reveal the underlying target trait ({trait_name}) implicitly
2. Be specific actions or habits, not abstract statements
3. Be diverse from each other
4. Each be one short sentence (under 20 words)

Output a JSON list of strings only, no explanation:
["behavior 1", "behavior 2", "behavior 3"]"""


def build_compact_reflection_prompt(
    current_reflection: str,
    candidate_reflections: List[str],
    behavior_samples: List[Dict[str, List[str]]],
    trait_name: str,
    trait_description: str,
    max_words: int = 50,
) -> str:
    """
    Compactness M-Step prompt (Algorithm 1, line 5 / Eq.3).
    Paper: "refine and select ê_{t-1} that can recover both the previous
    candidate ek and its corresponding behavior s_j^k"
    Goal: preserve shared behavior-driving patterns, remove noise.
    """
    pairs_text = ""
    for i, (ref, behaviors) in enumerate(zip(candidate_reflections, behavior_samples)):
        beh_str = "; ".join(behaviors.get("behaviors", []))
        pairs_text += f"\nCandidate {i+1}: {ref}\n  Implied behaviors: {beh_str}\n"

    return f"""You are optimizing a self-reflection for compactness.
Your goal is to preserve the shared trait-driving patterns across all candidates while removing redundant or irrelevant details.

Target trait: {trait_name}
Trait description: {trait_description}

Current global reflection:
{current_reflection}

Candidate reflections and their implied behaviors:
{pairs_text}

Please synthesize a single compact first-person self-reflection that:
1. Preserves the common behavior-driving patterns shared across candidates
2. Removes irrelevant background details (age, family, location, etc.)
3. Removes repeated expressions and stop words
4. Is generalizable across different tasks (questionnaires, creative writing, QA)
5. Uses first-person self-perceived experience format: "I [verb]..., e.g.: [concrete example]"
6. Stays under {max_words} words

Output only the compact reflection text, nothing else:"""


def build_response_sampling_prompt(
    reflection: str,
    task_prompt: str,
) -> str:
    """
    Evocativeness E-Step prompt (Algorithm 1, line 7).
    Paper: "sample {y_i^{j,t}} ~ p_{ê^{t-1}}(y|x_i)"
    The reflection is injected as a prefix to guide response generation.
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
    Evocativeness M-Step prompt (Algorithm 1, line 11 / Eq.5).
    Paper: "prompt the LLM to optimize the self-reflection, generate
    candidates, and select the top ones based on the score R2(e)"
    Compare high-scoring vs low-scoring responses to identify what works.
    """
    top_text = ""
    for i, r in enumerate(top_responses[:5]):
        top_text += f"\n[High-scoring response {i+1}] (score: {r['score']:.2f})\nTask: {r['task']}\nResponse: {r['response'][:300]}\n"

    bottom_text = ""
    for i, r in enumerate(bottom_responses[:5]):
        bottom_text += f"\n[Low-scoring response {i+1}] (score: {r['score']:.2f})\nTask: {r['task']}\nResponse: {r['response'][:300]}\n"

    return f"""You are optimizing a self-reflection to better elicit a target trait in a language model.
Analyze the high-scoring and low-scoring responses to understand what reflection patterns work.

Target trait: {trait_name}
Trait description: {trait_description}

Current compact reflection:
{compacted_reflection}

High-scoring responses (these effectively reflect the trait):
{top_text}

Low-scoring responses (these fail to reflect the trait):
{bottom_text}

Please generate {K} revised candidate self-reflections that:
1. Strengthen patterns that caused high trait scores
2. Avoid patterns that caused low trait scores
3. Do not overfit to a single task - be generalizable
4. Keep each reflection under {max_words} words
5. Use first-person self-reflection format: "I [verb]..., e.g.: [concrete example]"
6. Do NOT use explicit labels like "I am {trait_name}" or "I have high {trait_name}"
7. Do NOT include demographic information (age, gender, country, religion, family)

Output a JSON list of strings only:
["reflection 1", "reflection 2", ..., "reflection {K}"]"""


def build_candidate_scoring_prompt(
    candidate_reflection: str,
    trait_name: str,
    trait_description: str,
) -> str:
    """
    Judge prompt for evaluating how well a reflection captures a trait.
    Used in candidate ranking (Algorithm 1, lines 12-13).
    """
    return f"""You evaluate whether a self-reflection effectively captures a target trait.

Target trait: {trait_name}
Trait description: {trait_description}

Self-reflection to evaluate:
{candidate_reflection}

Score from 1 to 5:
1 = Does not capture the trait at all, or uses explicit labels
2 = Weakly captures the trait, mostly generic
3 = Moderately captures the trait, some good patterns
4 = Strongly captures the trait with concrete behavioral patterns
5 = Excellent trait capture, evocative and compact

Output JSON only:
{{"score": <1-5>, "reason": "brief explanation"}}"""


# ============================================================
# Response Parsing Utilities
# ============================================================

def extract_json_list(text: str) -> List[str]:
    """Extract a JSON list from LLM response text."""
    if not text or not text.strip():
        return []

    # Clean up common LLM artifacts
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Try to find JSON array in the text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(item).strip() for item in result if item and str(item).strip()]
        except json.JSONDecodeError:
            pass

    # Try parsing the entire text as JSON
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(item).strip() for item in result if item and str(item).strip()]
        # Handle JSON object with "candidates" key
        if isinstance(result, dict):
            for key in ["candidates", "reflections", "items", "results"]:
                if key in result and isinstance(result[key], list):
                    items = result[key]
                    extracted = []
                    for item in items:
                        if isinstance(item, dict):
                            # Try common keys for reflection text
                            for k in ["reflection", "text", "content", "value"]:
                                if k in item:
                                    extracted.append(str(item[k]).strip())
                                    break
                            else:
                                extracted.append(str(item).strip())
                        else:
                            extracted.append(str(item).strip())
                    return [x for x in extracted if x]
    except json.JSONDecodeError:
        pass

    # Fallback: extract numbered items (1. xxx, 2. xxx)
    items = re.findall(r'\d+\.\s*["\']?(.+?)["\']?\s*(?:\n|$)', text)
    if items and len(items) >= 2:
        return [item.strip().strip('"').strip("'") for item in items if item.strip()]

    # Fallback: extract quoted strings
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted and len(quoted) >= 2:
        return [q.strip() for q in quoted if q.strip()]

    # Last resort: split by newlines
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    cleaned = []
    for line in lines:
        # Remove leading numbers, bullets, etc.
        cleaned_line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
        if len(cleaned_line) > 10:
            cleaned.append(cleaned_line)
    return cleaned


def extract_json_score(text: str) -> Tuple[float, str]:
    """Extract score and reason from judge response."""
    if not text or not text.strip():
        return 3.0, "empty_response"

    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = data.get("score", None)
            reason = data.get("reason", "")
            if score is not None:
                return float(score), str(reason)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: extract first number between 1-5
    numbers = re.findall(r'\b([1-5])\b', text)
    if numbers:
        return float(numbers[0]), ""

    return 3.0, "parse_failed"


def extract_questionnaire_score(response: str, scale: int = 5, reverse: bool = False) -> Optional[float]:
    """
    Extract numerical score from questionnaire response.
    Paper: "trait evaluator q_ω is rule-based for questionnaires"
    """
    # Try to find a number in the response
    numbers = re.findall(r'\b(\d+)\b', response.strip())
    if numbers:
        raw = int(numbers[0])
        if 1 <= raw <= scale:
            if reverse:
                raw = scale + 1 - raw
            return (raw - 1) / (scale - 1)  # normalize to [0, 1]
    return None


# ============================================================
# Core Algorithm Functions
# ============================================================

def sample_behaviors(
    router,
    current_reflection: str,
    candidate_reflections: List[str],
    trait_name: str,
    trait_description: str,
    model_name: str,
    M1: int = 3,
    temperature: float = 0.9,
    max_tokens: int = 512,
) -> List[Dict[str, List[str]]]:
    """
    Compactness E-Step (Algorithm 1, lines 2-4).
    For each candidate reflection ek, sample M1 behaviors.
    Returns: list of dicts, each mapping candidate -> behaviors.
    """
    all_behaviors = []
    for cand in candidate_reflections:
        prompt = build_behavior_sampling_prompt(
            current_reflection=current_reflection,
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
        all_behaviors.append({
            "candidate": cand,
            "behaviors": behaviors[:M1],
        })
    return all_behaviors


def compact_reflection(
    router,
    current_reflection: str,
    candidate_reflections: List[str],
    behavior_samples: List[Dict],
    trait_name: str,
    trait_description: str,
    model_name: str,
    max_words: int = 50,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """
    Compactness M-Step (Algorithm 1, line 5 / Eq.3).
    Paper: "refine and select ê_{t-1} that can recover both the previous
    candidate ek and its corresponding behavior s_j^k"
    """
    prompt = build_compact_reflection_prompt(
        current_reflection=current_reflection,
        candidate_reflections=candidate_reflections,
        behavior_samples=behavior_samples,
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
    return responses[0].strip() if responses else current_reflection


def sample_task_responses(
    router,
    reflection: str,
    task_prompts: List[str],
    model_name: str,
    M2: int = 6,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> List[List[str]]:
    """
    Evocativeness E-Step (Algorithm 1, lines 6-8).
    Paper: "for each x_i, sample {y_i^{j,t}} ~ p_{ê^{t-1}}(y|x_i)"
    Returns: list of M2 responses per task.
    """
    all_responses = []
    for task in task_prompts:
        prompt = build_response_sampling_prompt(reflection=reflection, task_prompt=task)
        task_responses = []
        # Sample M2 responses
        for _ in range(M2):
            messages = [[{"role": "user", "content": prompt}]]
            responses = router.request_llm(
                conversations=messages,
                model=model_name,
                max_length=max_tokens,
                temperature=temperature,
            )
            if responses:
                task_responses.append(responses[0])
        all_responses.append(task_responses)
    return all_responses


def evaluate_responses(
    router,
    task_prompts: List[str],
    response_sets: List[List[str]],
    trait_name: str,
    trait_description: str,
    model_name: str,
    judge_model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> List[List[float]]:
    """
    Evocativeness E-Step (Algorithm 1, line 9).
    Paper: "Calculate q_ω(v|y_i^{j,t}, x_i) for each y_i^{j,t}"
    For closed-source models, use LLM-as-judge to approximate q_omega.
    Returns: list of scores per response per task.
    """
    if judge_model is None:
        judge_model = model_name

    all_scores = []
    for task_idx, (task, responses) in enumerate(zip(task_prompts, response_sets)):
        task_scores = []
        for resp in responses:
            # Use judge model to evaluate
            judge_prompt = build_candidate_scoring_prompt(
                candidate_reflection=resp,
                trait_name=trait_name,
                trait_description=trait_description,
            )
            messages = [[{"role": "user", "content": judge_prompt}]]
            judge_responses = router.request_llm(
                conversations=messages,
                model=judge_model,
                max_length=max_tokens,
                temperature=temperature,
            )
            score, _ = extract_json_score(judge_responses[0]) if judge_responses else (3.0, "")
            task_scores.append((score - 1) / 4.0)  # normalize to [0, 1]
        all_scores.append(task_scores)
    return all_scores


def compute_R2(
    response_scores: List[List[float]],
    response_probs: List[List[float]] = None,
) -> float:
    """
    Compute R2 score (Eq.5).
    Paper: R2(e) = (1/N) * sum_i sum_j p_e(y|x) * log q_omega(v|y,x)
    For closed-source models: use uniform weights (1/M2) as approximation.
    """
    all_weighted = []
    for task_scores in response_scores:
        for j, score in enumerate(task_scores):
            if response_probs is not None:
                prob = response_probs[len(all_weighted) // len(task_scores)][j]
            else:
                prob = 1.0 / max(len(task_scores), 1)  # uniform weight
            eps = 1e-6
            all_weighted.append(prob * np.log(score + eps))
    return np.mean(all_weighted) if all_weighted else 0.0


def generate_revised_candidates(
    router,
    compacted_reflection: str,
    task_prompts: List[str],
    response_sets: List[List[str]],
    response_scores: List[List[float]],
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
    Paper: "prompt the LLM to optimize the self-reflection, generate
    candidates, and select the top ones based on the score R2(e)"
    """
    # Collect all responses with scores
    scored_responses = []
    for task_idx, (task, responses, scores) in enumerate(zip(task_prompts, response_sets, response_scores)):
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
    judge_model: str = None,
    M2: int = 2,
    beta: float = 1.0,
    gamma: float = 0.3,
    alpha_len: float = 0.1,
    alpha_dup: float = 0.1,
    max_words: int = 50,
) -> List[Reflection]:
    """
    Candidate ranking (Algorithm 1, lines 12-13).
    Paper: "Calculate R2(e_k^t) for each e_k^t in E_t"
    For each candidate, sample a few responses and compute final score.
    """
    if judge_model is None:
        judge_model = model_name

    ranked = []
    for cand_text in candidates:
        # Sample a few validation responses
        resp_sets = sample_task_responses(
            router=router,
            reflection=cand_text,
            task_prompts=validation_tasks[:3],  # use 3 tasks for efficiency
            model_name=model_name,
            M2=M2,
            temperature=0.7,
        )

        # Evaluate responses
        scores = evaluate_responses(
            router=router,
            task_prompts=validation_tasks[:3],
            response_sets=resp_sets,
            trait_name=trait_name,
            trait_description=trait_description,
            model_name=model_name,
            judge_model=judge_model,
            temperature=0.0,
        )

        # Compute metrics
        all_scores = [s for task_scores in scores for s in task_scores]
        evocativeness = np.mean(all_scores) if all_scores else 0.0
        stability = 1.0 - min(np.std(all_scores), 0.5) if len(all_scores) > 1 else 0.5

        # Length penalty
        words = cand_text.split()
        length_penalty = max(0, (len(words) - max_words) / max_words)

        # Redundancy penalty
        unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
        redundancy_penalty = 1.0 - unique_ratio

        # Final score
        final_score = (
            beta * evocativeness
            + gamma * stability
            - alpha_len * length_penalty
            - alpha_dup * redundancy_penalty
        )

        ref = Reflection(
            text=cand_text,
            source="revise",
            scores={
                "final_score": final_score,
                "evocativeness": evocativeness,
                "stability": stability,
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
    judge_model: str = None,
    K: int = 10,
    M1: int = 3,
    M2: int = 6,
    max_words: int = 50,
    beta: float = 1.0,
    gamma: float = 0.3,
    alpha_len: float = 0.1,
    alpha_dup: float = 0.1,
) -> Dict:
    """
    Run one full IROTE iteration (Algorithm 1, lines 1-14).

    Returns dict with:
        - compacted_reflection
        - behavior_samples
        - response_sets
        - response_scores
        - revised_candidates
        - ranked_candidates
        - best_reflection
    """
    print(f"\n{'='*60}")
    print(f"IROTE Iteration {iteration}")
    print(f"{'='*60}")

    # Step 1: Compactness E-Step (Algorithm 1, lines 2-4)
    print(f"\n[Step 1] Compactness E-Step: Sampling {M1} behaviors per candidate...")
    behavior_samples = sample_behaviors(
        router=router,
        current_reflection=current_reflection,
        candidate_reflections=candidate_reflections,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=model_name,
        M1=M1,
    )
    print(f"  Generated behaviors for {len(behavior_samples)} candidates")

    # Step 2: Compactness M-Step (Algorithm 1, line 5 / Eq.3)
    print(f"\n[Step 2] Compactness M-Step: Compacting reflection...")
    compacted = compact_reflection(
        router=router,
        current_reflection=current_reflection,
        candidate_reflections=candidate_reflections,
        behavior_samples=behavior_samples,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=model_name,
        max_words=max_words,
    )
    print(f"  Compacted reflection ({len(compacted.split())} words): {compacted[:100]}...")

    # Step 3: Evocativeness E-Step (Algorithm 1, lines 6-8)
    print(f"\n[Step 3] Evocativeness E-Step: Sampling {M2} responses per task...")
    response_sets = sample_task_responses(
        router=router,
        reflection=compacted,
        task_prompts=optimize_tasks,
        model_name=model_name,
        M2=M2,
    )

    # Step 4: Evaluate responses (Algorithm 1, line 9)
    print(f"\n[Step 4] Evaluating responses with q_omega...")
    response_scores = evaluate_responses(
        router=router,
        task_prompts=optimize_tasks,
        response_sets=response_sets,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=model_name,
        judge_model=judge_model,
    )

    # Compute R2 for compacted reflection
    r2_score = compute_R2(response_scores)
    avg_score = np.mean([s for ts in response_scores for s in ts]) if response_scores else 0
    print(f"  R2 score: {r2_score:.4f}, Avg trait score: {avg_score:.4f}")

    # Step 5: Evocativeness M-Step (Algorithm 1, line 11 / Eq.5)
    print(f"\n[Step 5] Evocativeness M-Step: Generating {K} revised candidates...")
    revised_candidates = generate_revised_candidates(
        router=router,
        compacted_reflection=compacted,
        task_prompts=optimize_tasks,
        response_sets=response_sets,
        response_scores=response_scores,
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=model_name,
        K=K,
        max_words=max_words,
    )
    print(f"  Generated {len(revised_candidates)} candidates")

    # Step 6: Rank candidates (Algorithm 1, lines 12-13)
    print(f"\n[Step 6] Ranking candidates...")
    ranked = rank_candidates(
        router=router,
        candidates=revised_candidates,
        validation_tasks=optimize_tasks[:5],  # use subset for efficiency
        trait_name=trait_name,
        trait_description=trait_description,
        model_name=model_name,
        judge_model=judge_model,
        M2=2,
        beta=beta,
        gamma=gamma,
        alpha_len=alpha_len,
        alpha_dup=alpha_dup,
        max_words=max_words,
    )

    if ranked:
        best = ranked[0]
    else:
        # Fallback: use compacted reflection with default scores
        best = Reflection(
            text=compacted,
            iteration=iteration,
            source="compact",
            scores={
                "final_score": r2_score,
                "evocativeness": avg_score,
                "stability": 0.5,
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
        "behavior_samples": behavior_samples,
        "response_sets": response_sets,
        "response_scores": response_scores,
        "r2_score": r2_score,
        "avg_trait_score": avg_score,
        "revised_candidates": [r.text for r in ranked],
        "ranked_candidates": ranked,
        "best_reflection": best,
    }
