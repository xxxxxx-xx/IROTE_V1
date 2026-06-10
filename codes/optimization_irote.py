"""
IROTE Optimization - Paper-Aligned Implementation
==================================================
Fixes applied:
1. Separate target_model and judge_model
2. PMI-based compactness (ProbabilityEstimator)
3. Confidence-weighted evocativeness
4. Complete trait descriptions and open task templates
5. Connected evaluation pipeline
"""

import random
random.seed(312)
import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.router import ModelRouter
from irote_core import (
    Reflection,
    run_irote_iteration,
    TRAIT_DESCRIPTIONS,
    extract_questionnaire_score,
)


# ============================================================
# Open Task Templates (Complete for all traits)
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
        "How do you maintain purity in your life?",
        "Describe a time when you avoided something you considered unclean or wrong.",
        "Write about what moral purity means to you.",
    ],
}


def load_optimize_tasks(evaluation_system: str, target_trait: str, questionnaires: dict) -> List[Dict]:
    """Load optimization tasks (questionnaire items for the target trait)."""
    tasks = []
    opt_questionnaires = {
        "personality": ["BFI"],
        "value": ["PVQ21", "PVQ-RR"],
        "moral": ["MFQ-1"],
    }

    q_names = opt_questionnaires.get(evaluation_system, ["BFI"])

    for q_name in q_names:
        q_data = questionnaires.get(q_name)
        if q_data is None:
            continue

        categories = {cat["cat_name"].lower(): cat["cat_questions"] for cat in q_data["categories"]}
        if target_trait.lower() not in categories:
            continue

        trait_idxes = categories[target_trait.lower()]
        questions_map = q_data["questions"]
        scale = q_data["scale"] - 1
        reverse_list = q_data.get("reverse", [])

        for q_idx in trait_idxes:
            q_text = questions_map.get(str(q_idx), "")
            if not q_text:
                continue

            is_reverse = q_idx in reverse_list
            inner_setting = q_data.get("inner_setting_single", "")
            prompt_single = q_data.get("prompt_single", "")

            task_prompt = f"{inner_setting}\n\n{prompt_single}\n\nStatement: {q_text}"

            tasks.append({
                "task_id": f"{q_name}_{q_idx}",
                "task_type": "questionnaire",
                "prompt": task_prompt,
                "reverse": is_reverse,
                "scale": scale,
                "trait": target_trait,
            })

    return tasks


def load_open_tasks(target_trait: str, n_tasks: int = 10) -> List[Dict]:
    """Generate open-ended tasks for evocativeness evaluation."""
    templates = OPEN_TASK_TEMPLATES.get(target_trait.lower(), [])
    if not templates:
        # Fallback to generic tasks
        templates = [
            f"Write a short paragraph about how you would approach a situation relevant to {target_trait}.",
            f"Describe a time when you demonstrated {target_trait}.",
            f"How do you view the importance of {target_trait} in daily life?",
            f"Write about a decision you made that reflects {target_trait}.",
            f"Describe how {target_trait} influences your relationships with others.",
        ]
    selected = random.sample(templates, min(n_tasks, len(templates)))

    return [
        {
            "task_id": f"open_{i}",
            "task_type": "open_generation",
            "prompt": t,
            "reverse": False,
            "trait": target_trait,
        }
        for i, t in enumerate(selected)
    ]


def run_irote_optimization(
    args,
    target_trait: str,
    router: ModelRouter,
):
    """Full IROTE optimization loop (Algorithm 1)."""
    output_dir = os.path.join(args.output_dir, target_trait)
    os.makedirs(output_dir, exist_ok=True)

    # Load questionnaires
    q_path = os.path.join(os.path.dirname(__file__), "data", "questionnaires.json")
    with open(q_path, "r") as f:
        questionnaires = {q["name"]: q for q in json.load(f)}

    # Get trait description
    trait_desc = TRAIT_DESCRIPTIONS.get(target_trait.lower(), f"The trait of {target_trait}")

    print(f"\n{'='*70}")
    print(f"IROTE Optimization: {target_trait}")
    print(f"Target Model: {args.target_model}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Iterations: {args.max_iteration}, K={args.K}, M1={args.M1}, M2={args.M2}, β={args.beta}")
    print(f"{'='*70}")

    # Load tasks
    questionnaire_tasks = load_optimize_tasks(args.evaluation_system, target_trait, questionnaires)
    open_tasks = load_open_tasks(target_trait, n_tasks=args.n_open_tasks)
    all_tasks = questionnaire_tasks + open_tasks

    print(f"\nLoaded {len(questionnaire_tasks)} questionnaire tasks + {len(open_tasks)} open tasks")

    # Initialize reflections from CSV
    csv_path = os.path.join(os.path.dirname(__file__), "data", "reflection_data",
                            f"{args.evaluation_system}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        trait_reflections = df[df["dimension"].str.lower() == target_trait.lower()]["sentence"].tolist()
    else:
        trait_reflections = []

    if len(trait_reflections) < args.K:
        print(f"Warning: Only {len(trait_reflections)} initial reflections (need {args.K}). Generating more...")
        from irote_core import extract_json_list
        gen_prompt = f"""Generate {args.K} diverse first-person self-reflections for trait: {target_trait}.
Description: {trait_desc}

Format: "I [verb]..., e.g.: [concrete example]"
Under 50 words each. No explicit labels like "I am {target_trait}". No demographics.

Output JSON list: ["reflection 1", ...]"""
        messages = [[{"role": "user", "content": gen_prompt}]]
        responses = router.request_llm(
            conversations=messages,
            model=args.target_model,
            max_length=2048,
            temperature=0.8,
        )
        generated = extract_json_list(responses[0]) if responses else []
        trait_reflections.extend(generated)

    candidate_texts = trait_reflections[:args.K]
    print(f"Initial reflections: {len(candidate_texts)}")

    # Task prompts
    task_prompts = [t["prompt"] for t in all_tasks]

    # Main optimization loop
    best_reflection = None
    best_score = -float("inf")
    history = []

    for t in range(1, args.max_iteration + 1):
        current_reflection = best_reflection if best_reflection else "\n".join(
            [f"{i+1}. {r}" for i, r in enumerate(candidate_texts[:5])]
        )

        result = run_irote_iteration(
            router=router,
            iteration=t,
            current_reflection=current_reflection,
            candidate_reflections=candidate_texts,
            optimize_tasks=task_prompts,
            trait_name=target_trait,
            trait_description=trait_desc,
            target_model=args.target_model,
            judge_model=args.judge_model,
            K=args.K,
            M1=args.M1,
            M2=args.M2,
            beta=args.beta,
            max_words=args.words_limit,
        )

        # Update best
        new_best = result["best_reflection"]
        new_best_score = new_best.scores.get("final_score", 0)
        if new_best_score > best_score:
            best_score = new_best_score
            best_reflection = new_best.text

        # Update candidates
        if result["ranked_candidates"]:
            candidate_texts = [r.text for r in result["ranked_candidates"][:args.K]]

        # Save iteration
        iter_save = {
            "iteration": t,
            "current_reflection": current_reflection,
            "compacted_reflection": result["compacted_reflection"],
            "r2_score": result["r2_score"],
            "avg_q_score": result["avg_q_score"],
            "best_reflection": new_best.text,
            "best_score": new_best.scores,
            "n_candidates": len(result["ranked_candidates"]),
        }
        history.append(iter_save)

        with open(os.path.join(output_dir, f"iter_{t}.json"), "w") as f:
            json.dump(iter_save, f, indent=2, ensure_ascii=False)

        print(f"\nIteration {t} Summary:")
        print(f"  R2: {result['r2_score']:.4f}, Avg q_ω: {result['avg_q_score']:.4f}")
        print(f"  Best score: {new_best_score:.4f}, Global best: {best_score:.4f}")

        # Early stopping
        if t >= 3 and len(history) >= 2:
            prev = history[-2]["best_score"].get("final_score", 0)
            curr = history[-1]["best_score"].get("final_score", 0)
            if curr - prev < 0.01:
                print(f"\nEarly stopping: improvement < 0.01")
                break

    # Save final result
    final = {
        "trait": target_trait,
        "target_model": args.target_model,
        "judge_model": args.judge_model,
        "params": {"K": args.K, "M1": args.M1, "M2": args.M2, "beta": args.beta, "T": args.max_iteration},
        "iterations": len(history),
        "best_reflection": best_reflection,
        "best_score": best_score,
        "history": history,
    }
    with open(os.path.join(output_dir, "final_reflection.json"), "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Optimization Complete!")
    print(f"Best reflection (score={best_score:.4f}):")
    print(f"{best_reflection}")
    print(f"Results: {output_dir}")
    print(f"{'='*70}")

    return final


def main():
    parser = argparse.ArgumentParser(description="IROTE Optimization")

    # Model settings (separate target and judge)
    parser.add_argument("--target_model", type=str, default="MiMo-v2.5-Pro",
                        help="Model being optimized (generates responses)")
    parser.add_argument("--judge_model", type=str, default="MiMo-v2.5-Pro",
                        help="Model evaluating responses (judge). Use different model to reduce self-evaluation bias")

    # Trait
    parser.add_argument("--evaluation_system", type=str, default="personality",
                        choices=["value", "personality", "moral"])
    parser.add_argument("--specific_traits", type=str, default="extraversion")

    # IROTE hyperparameters (paper defaults)
    parser.add_argument("--max_iteration", type=int, default=5)
    parser.add_argument("--K", type=int, default=10, help="Candidate reflections (paper: 10)")
    parser.add_argument("--M1", type=int, default=3, help="Behaviors per candidate (paper: 3)")
    parser.add_argument("--M2", type=int, default=6, help="Responses per task (paper: 6)")
    parser.add_argument("--beta", type=float, default=1.0, help="Evocativeness weight (paper: 1.0)")
    parser.add_argument("--words_limit", type=int, default=50, help="Max reflection words")
    parser.add_argument("--n_open_tasks", type=int, default=5, help="Number of open-ended tasks")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/irote_results")

    args = parser.parse_args()

    # Compatibility: --model_name sets both target and judge
    if hasattr(args, 'model_name') and args.model_name:
        args.target_model = args.model_name
        args.judge_model = args.model_name

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Initializing target model: {args.target_model}")
    print(f"Initializing judge model: {args.judge_model}")
    router = ModelRouter(
        model_names=[args.target_model, args.judge_model],
        temperature=1.0,
        top_p=0.95,
        max_model_len=1024,
    )

    traits = [t.strip() for t in args.specific_traits.split(",")]
    for trait in traits:
        run_irote_optimization(args, trait, router)


if __name__ == "__main__":
    main()
