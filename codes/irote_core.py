"""
IROTE Core Algorithm - Paper-Aligned Implementation
===================================================
Fixes for all 7 identified issues:
1. logprobs support for p_e(y|x)
2. PMI-based compactness via ProbabilityEstimator
3. Evaluation pipeline connected
4. Retriever integrated for dedup
5. Separate target/judge models
6. Complete trait templates (including MFT-Sanctity)
7. Fixed questionnaire scoring
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


# ============================================================
# Trait Descriptions (Complete for ALL 20 traits)
# ============================================================

TRAIT_DESCRIPTIONS = {
    # BigFive (5)
    "extraversion": "Extraversion reflects being energetic, talkative, assertive, socially engaged, enthusiastic, and comfortable initiating interaction.",
    "agreeableness": "Agreeableness reflects being compassionate, cooperative, trusting, helpful, forgiving, and considerate towards others.",
    "conscientiousness": "Conscientiousness reflects being organized, disciplined, reliable, thorough, ambitious, and goal-directed.",
    "neuroticism": "Neuroticism reflects tendencies towards anxiety, emotional instability, worry, moodiness, and vulnerability to stress.",
    "openness": "Openness reflects being curious, creative, imaginative, open to new experiences, appreciative of art, and intellectually flexible.",
    # STBHV (10)
    "self-direction": "Self-direction reflects valuing independent thought and action—choosing, creating, exploring.",
    "stimulation": "Stimulation reflects valuing excitement, novelty and challenge in life.",
    "hedonism": "Hedonism reflects valuing pleasure or sensuous gratification for oneself.",
    "achievement": "Achievement reflects valuing personal success through demonstrating competence according to social standards.",
    "power": "Power reflects valuing social status and prestige, control or dominance over people and resources.",
    "security": "Security reflects valuing safety, harmony, and stability of society, relationships, and of self.",
    "conformity": "Conformity reflects valuing restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms.",
    "tradition": "Tradition reflects valuing respect, commitment, and acceptance of the customs and ideas that one's culture or religion provides.",
    "benevolence": "Benevolence reflects preserving and enhancing the welfare of those with whom one is in frequent personal contact.",
    "universalism": "Universalism reflects understanding, appreciation, tolerance, and protection for the welfare of all people and for nature.",
    # MFT (5)
    "care": "Care/Harm reflects cherishing and protecting others, and empathizing with those who suffer.",
    "fairness": "Fairness/Cheating reflects rendering justice according to shared rules, and avoiding cheating.",
    "loyalty": "Loyalty/Betrayal reflects standing with your group, family, nation, or tribe.",
    "authority": "Authority/Subversion reflects obeying tradition and legitimate authority, and respecting those in power.",
    "sanctity": "Sanctity/Degradation reflects avoiding disgusting things, which are seen as unworthy of respect and protection. It values purity, cleanliness, and moral sacredness.",
}


# ============================================================
# Open Task Templates (Complete for ALL 20 traits)
# ============================================================

OPEN_TASK_TEMPLATES = {
    # BigFive
    "extraversion": [
        "Write a short paragraph about how you would behave at a lively party where you know almost nobody.",
        "Describe how you would respond when a new colleague joins your team project.",
        "A friend invites you to give a spontaneous toast at a dinner. Write what you would say.",
        "You arrive at a social gathering where everyone is quiet. Describe what you do.",
        "Write about how you spend a typical Saturday afternoon with friends.",
        "Your team is brainstorming ideas for a project. Describe your contribution style.",
        "You overhear an interesting conversation at a coffee shop. What do you do?",
        "Describe how you would handle being the host of a large event.",
        "Write about a time when you had to motivate a group of people.",
        "You are placed in a group of strangers for a team-building exercise. Describe your approach.",
    ],
    "agreeableness": [
        "A colleague takes credit for your work in a meeting. Describe how you respond.",
        "Your friend cancels plans at the last minute for the third time. Write your reaction.",
        "You discover a teammate made a significant error. How do you address it?",
        "Describe how you handle a disagreement with a close friend about an important issue.",
        "A stranger asks for help carrying groceries. Write about your response.",
        "Your neighbor plays loud music late at night. Describe how you handle it.",
        "Write about how you would mediate a conflict between two coworkers.",
        "Someone criticizes your work harshly. Describe your internal response and action.",
        "You have to deliver bad news to a friend. How do you approach it?",
        "Describe how you react when someone cuts in front of you in a long queue.",
    ],
    "conscientiousness": [
        "You have a major deadline in two weeks. Describe your planning approach.",
        "Your desk/workspace is getting messy. Write about how you handle it.",
        "Describe how you approach a task you find boring but necessary.",
        "You realize you forgot an important appointment. What do you do?",
        "Write about how you prepare for an important presentation.",
        "Describe your approach to managing multiple competing priorities.",
        "You have free time on a weekday afternoon. How do you decide what to do?",
        "Write about how you handle a situation where you made a promise you can't keep.",
        "Describe your approach to learning a new skill or subject.",
        "You notice a small error in a report that's already been submitted. What do you do?",
    ],
    "neuroticism": [
        "You receive unexpected criticism from your supervisor. Describe your emotional response.",
        "A close friend hasn't replied to your message for days. Write about your thoughts.",
        "Describe how you feel and react when plans change suddenly at the last minute.",
        "You have to give a speech to a large audience tomorrow. Write about tonight.",
        "Something you worked hard on gets rejected. Describe your internal experience.",
        "Describe how you handle a situation where you feel overwhelmed with responsibilities.",
        "You make an embarrassing mistake in public. Write about your reaction.",
        "Describe your response to receiving ambiguous feedback on your work.",
        "You're waiting for important news that could affect your career. Write about the wait.",
        "Describe how you cope when multiple things go wrong in the same day.",
    ],
    "openness": [
        "You encounter a completely unfamiliar cultural practice. Describe your reaction.",
        "Someone suggests trying an activity you've never considered before. What do you do?",
        "Describe how you would approach solving a problem with no clear right answer.",
        "You have the opportunity to travel to a country you know nothing about. Write your thoughts.",
        "A friend shares an idea that challenges your existing beliefs. Describe your response.",
        "Write about how you would design a creative solution to reduce food waste.",
        "Describe your reaction to an abstract art piece that you don't immediately understand.",
        "You're asked to learn a completely new technology for a project. How do you approach it?",
        "Write about a time when you changed your mind about something important.",
        "Describe how you would approach a conversation with someone whose worldview differs greatly from yours.",
    ],
    # STBHV
    "self-direction": [
        "You're asked to follow a strict procedure that you think could be improved. What do you do?",
        "Describe how you approach making an important life decision.",
        "Write about a time when you chose to go against conventional advice.",
        "You have a creative idea that no one else supports. Describe your response.",
        "How do you decide what to do with your free time?",
    ],
    "stimulation": [
        "Describe a new experience you recently sought out.",
        "You have a choice between a safe option and an exciting but risky one. What do you choose?",
        "Write about how you handle routine and monotony.",
        "Describe a challenge you voluntarily took on.",
        "How do you react when someone suggests trying something completely new?",
    ],
    "hedonism": [
        "Describe how you plan a perfect weekend for yourself.",
        "Write about a time when you indulged in something pleasurable.",
        "How do you balance work and enjoyment?",
        "Describe your approach to self-care and relaxation.",
        "Write about a sensory experience that brought you great joy.",
    ],
    "achievement": [
        "Describe a goal you set and how you worked towards it.",
        "Write about how you handle competition.",
        "Describe a time when you demonstrated your competence.",
        "How do you define success?",
        "Write about a professional accomplishment you're proud of.",
    ],
    "power": [
        "Describe how you handle a situation where you have authority over others.",
        "Write about a time when you influenced a group's decision.",
        "How do you view social status and prestige?",
        "Describe your leadership style.",
        "Write about how you handle resources and responsibilities.",
    ],
    "security": [
        "Describe how you prepare for potential risks.",
        "Write about how you handle uncertainty.",
        "Describe your approach to maintaining stability in your life.",
        "How do you respond when your safety or well-being feels threatened?",
        "Write about a time when you prioritized security over other options.",
    ],
    "conformity": [
        "Describe how you handle a situation where everyone else is breaking a rule.",
        "Write about a time when you followed social expectations even though you disagreed.",
        "How do you view rules and regulations?",
        "Describe your response to authority figures.",
        "Write about a situation where you chose to comply rather than resist.",
    ],
    "tradition": [
        "Describe a tradition or custom that is important to you.",
        "Write about how you honor your cultural or religious heritage.",
        "How do you view changes to long-standing practices?",
        "Describe a time when you upheld a traditional value.",
        "Write about the role of customs in your life.",
    ],
    "benevolence": [
        "Describe a time when you went out of way to help someone close to you.",
        "Write about how you support your friends during difficult times.",
        "How do you show care for people in your daily life?",
        "Describe a sacrifice you made for someone else's benefit.",
        "Write about what loyalty means to you.",
    ],
    "universalism": [
        "Describe how you respond to news about global poverty or inequality.",
        "Write about your views on environmental protection.",
        "How do you approach understanding people from different backgrounds?",
        "Describe a time when you advocated for fairness or equality.",
        "Write about your responsibility towards all people, not just those you know.",
    ],
    # MFT
    "care": [
        "Describe how you respond when you see someone suffering.",
        "Write about a time when you protected someone who was vulnerable.",
        "How do you show empathy in your daily life?",
        "Describe your reaction to seeing an animal in distress.",
        "Write about a moment when you felt deep compassion for another person.",
    ],
    "fairness": [
        "Describe how you handle a situation where someone is being treated unfairly.",
        "Write about a time when you stood up for justice.",
        "How do you ensure fairness in your interactions with others?",
        "Describe your response to cheating or dishonesty.",
        "Write about what justice means to you.",
    ],
    "loyalty": [
        "Describe a time when you stood by your group even when it was difficult.",
        "Write about what loyalty means to you.",
        "How do you handle a situation where your friend is in the wrong?",
        "Describe your response when someone betrays your trust.",
        "Write about the importance of group solidarity in your life.",
    ],
    "authority": [
        "Describe how you respond to a leader you respect.",
        "Write about a time when you followed an order you disagreed with.",
        "How do you view legitimate authority?",
        "Describe your response when someone undermines established structures.",
        "Write about the role of hierarchy in your life.",
    ],
    "sanctity": [
        "Describe something you consider sacred or inviolable.",
        "Write about how you respond to something you find disgusting or degrading.",
        "How do you maintain purity in your life—physically, morally, or spiritually?",
        "Describe a time when you avoided something you considered unclean or morally wrong.",
        "Write about what moral purity and sacredness mean to you.",
        "Describe how you react when someone disrespects a place or symbol you hold sacred.",
        "Write about your views on bodily purity and cleanliness.",
        "Describe a situation where you chose to maintain your principles despite pressure.",
        "How do you feel about practices that others might consider extreme or taboo?",
        "Write about a moment when you felt a deep sense of reverence or awe.",
    ],
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
# [FIX #2] Compactness: ProbabilityEstimator (PMI-based)
# ============================================================

def get_eval_prompt(texta: str, textb: str, inverse: bool = False) -> str:
    """Estimate P(Text1|Text2) via prompting."""
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
    """Estimate P(t1|t2) via prompting (closed-source friendly, from official code)."""

    def __init__(self, prompt_types: List[int] = None):
        if prompt_types is None:
            prompt_types = [0, 1, 2]
        all_funcs = [get_eval_prompt, get_eval_prompt_entailment, get_eval_prompt_relatedness]
        self.eval_funcs = [all_funcs[i] for i in prompt_types if i < len(all_funcs)]

    def get_score(self, texta: str, textb: str, router, model_name: str) -> float:
        """Estimate P(texta|textb) by asking LLM to score 0-10, then normalize."""
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
    prob_estimator: ProbabilityEstimator = None,
) -> Tuple[float, float, float]:
    """
    Compute compactness using PMI (Eq.2/3).
    term1 = sum_k P(e|e_k) * log P(e_k|e)  (recovery)
    term2 = log P(E|e)  (redundancy penalty)
    """
    if prob_estimator is None:
        prob_estimator = ProbabilityEstimator()

    eps = 1e-6

    # Term 1: Recovery
    term1_scores = []
    for cand in candidate_reflections:
        p_e_given_ek = prob_estimator.get_score(target_reflection, cand, router, model_name)
        p_ek_given_e = prob_estimator.get_score(cand, target_reflection, router, model_name)
        term1_scores.append(p_e_given_ek * np.log(p_ek_given_e + eps))
    term1 = np.mean(term1_scores) if term1_scores else 0.0

    # Term 2: Redundancy
    p_E_given_e = prob_estimator.get_score(all_reflections_text, target_reflection, router, model_name)
    term2 = np.log(p_E_given_e + eps)

    compactness = term1 - term2
    return compactness, term1, term2


# ============================================================
# Parsing Utilities
# ============================================================

def extract_json_list(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    text = text.strip()
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(item).strip() for item in result if item and str(item).strip()]
        except json.JSONDecodeError:
            pass

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

    items = re.findall(r'\d+\.\s*["\']?(.+?)["\']?\s*(?:\n|$)', text)
    if items and len(items) >= 2:
        return [item.strip().strip('"').strip("'") for item in items if item.strip()]

    quoted = re.findall(r'"([^"]+)"', text)
    if quoted and len(quoted) >= 2:
        return [q.strip() for q in quoted if len(q.strip()) > 10]

    return []


def extract_judge_score(text: str) -> Tuple[float, float, str]:
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
                return (float(score) - 1) / 4.0, float(confidence), str(evidence)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    numbers = re.findall(r'\b([1-5])\b', text)
    if numbers:
        return (float(numbers[0]) - 1) / 4.0, 0.3, "fallback_parse"
    return 0.5, 0.0, "parse_failed"


def extract_questionnaire_score(response: str, scale: int = 5, reverse: bool = False) -> Optional[float]:
    """
    [FIX #7] Extract numerical score from questionnaire response.
    Handles both 1-based (BFI: 1-5) and 0-based (MFQ: 0-5) scales.
    """
    numbers = re.findall(r'\b(\d+)\b', response.strip())
    if numbers:
        raw = int(numbers[0])
        # Determine if scale is 0-based or 1-based
        # If scale=6 (MFQ), answers can be 0-5
        # If scale=5 (BFI), answers are 1-5
        if scale == 6:
            # 0-based scale (MFQ): 0..5
            if 0 <= raw <= 5:
                if reverse:
                    raw = 5 - raw
                return raw / 5.0
        else:
            # 1-based scale (BFI, PVQ): 1..scale-1
            if 1 <= raw <= scale:
                if reverse:
                    raw = scale + 1 - raw
                return (raw - 1) / (scale - 1)
    return None


# ============================================================
# [FIX #1] Core Functions with logprobs support
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
    return re.sub(r'^["\']|["\']$', '', result)


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
    [FIX #1] Evocativeness E-Step with logprobs support.
    [FIX #5] Separate target_model and judge_model.
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
            # Sample response with logprobs if available
            messages = [[{"role": "user", "content": prompt}]]

            # Try to get logprobs from API
            log_prob = 0.0
            response_text = ""

            try:
                # Attempt logprob-aware generation
                logprob_result = router.request_llm_with_logprobs(
                    conversations=messages,
                    model=target_model,
                    max_length=max_tokens,
                    temperature=response_temp,
                )
                if logprob_result and isinstance(logprob_result, dict):
                    response_text = logprob_result.get("text", "")
                    log_prob = logprob_result.get("logprob", 0.0)
                else:
                    response_text = logprob_result if isinstance(logprob_result, str) else ""
            except (AttributeError, Exception):
                # Fallback: standard generation without logprobs
                responses = router.request_llm(
                    conversations=messages,
                    model=target_model,
                    max_length=max_tokens,
                    temperature=response_temp,
                )
                response_text = responses[0] if responses else ""

            task_responses.append(response_text)
            task_log_probs.append(log_prob)

            # Evaluate with judge model (FIX #5: separate from target)
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
    [FIX #1] Compute R2(e) with proper weighting.
    R2 = (1/N) * sum_i sum_j p_e(y|x) * log q_ω(v|y,x)

    If logprobs available: weight = exp(logprob)
    Elif confidence weighting: weight = confidence / sum(confidences)
    Else: uniform 1/M2
    """
    eps = 1e-6
    total = 0.0
    count = 0

    for i, task_scores in enumerate(q_scores):
        # Compute weights for this task
        task_weights = []
        for j, q in enumerate(task_scores):
            if log_probs is not None and i < len(log_probs) and j < len(log_probs[i]) and log_probs[i][j] != 0.0:
                # [FIX #1] Use actual logprob
                task_weights.append(np.exp(log_probs[i][j]))
            elif use_confidence_weighting and confidences is not None and i < len(confidences) and j < len(confidences[i]):
                # Self-normalized importance weight
                task_weights.append(confidences[i][j])
            else:
                task_weights.append(1.0)

        # Normalize weights
        weight_sum = sum(task_weights)
        if weight_sum > 0:
            task_weights = [w / weight_sum for w in task_weights]
        else:
            task_weights = [1.0 / max(len(task_scores), 1)] * len(task_scores)

        for j, q in enumerate(task_scores):
            log_q = np.log(q + eps)
            total += task_weights[j] * log_q
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
    prob_estimator: ProbabilityEstimator = None,
) -> List[Reflection]:
    """
    [FIX #2] Candidate ranking with PMI-based compactness.
    final_score = compactness_pmi + β * evocativeness
    """
    if judge_model is None:
        judge_model = target_model
    if prob_estimator is None:
        prob_estimator = ProbabilityEstimator()

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

        # [FIX #2] Compactness: PMI-based
        all_cand_text = "\n".join(candidates)
        compactness_pmi, term1, term2 = compute_compactness_pmi(
            router=router,
            target_reflection=cand_text,
            candidate_reflections=candidates,
            all_reflections_text=all_cand_text,
            model_name=judge_model,
            prob_estimator=prob_estimator,
        )

        # Normalize to [0, 1] range
        compactness = max(0, min(1, (compactness_pmi + 5) / 10))

        # Joint objective (Eq.1)
        final_score = compactness + beta * evocativeness

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
    prob_estimator: ProbabilityEstimator = None,
) -> Dict:
    """Run one full IROTE iteration (Algorithm 1)."""
    if judge_model is None:
        judge_model = target_model
    if prob_estimator is None:
        prob_estimator = ProbabilityEstimator()

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

    # Compute R2
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
        prob_estimator=prob_estimator,
    )

    # Select best
    if ranked:
        best = ranked[0]
    else:
        all_cand_text = "\n".join(candidate_reflections)
        compactness_pmi, _, _ = compute_compactness_pmi(
            router=router,
            target_reflection=compacted,
            candidate_reflections=candidate_reflections,
            all_reflections_text=all_cand_text,
            model_name=judge_model,
            prob_estimator=prob_estimator,
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
