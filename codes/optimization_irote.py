"""
IROTE Full Optimization Script
===============================
Integrates irote_core algorithm with existing evaluation framework.
Implements the complete Algorithm 1 from the paper.

Usage:
    E:\\anaconda\\envs\\cottonagent\\python.exe optimization_irote.py \\
        --model_name MiMo-v2.5-Pro \\
        --evaluation_system personality \\
        --specific_traits extraversion \\
        --max_iteration 5
"""

import random
random.seed(312)
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PATH_TO_SIMCSE_MODEL
from models.router import ModelRouter
from eval_utils import Evaluator
from eval_utils.controller import Controller
from tools import Retriever

from irote_core import (
    Reflection,
    sample_behaviors,
    compact_reflection,
    sample_task_responses,
    evaluate_responses,
    compute_R2,
    generate_revised_candidates,
    rank_candidates,
    extract_questionnaire_score,
    run_irote_iteration,
)


def load_optimize_tasks(evaluation_system: str, target_trait: str, questionnaires: dict) -> List[Dict]:
    """
    Load optimization tasks (questionnaire items for the target trait).
    Paper: "Questionnaires marked with * are used for reflection optimization"
    For BigFive: BFI*; For STBHV: PVQ21*, PVQ-RR*; For MFT: MFQ-1*
    """
    tasks = []

    # Select optimization questionnaires based on system
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
    """
    Generate open-ended tasks for evocativeness evaluation.
    Paper uses downstream tasks like creative writing, controversial QA, etc.
    For initial reproduction, we use template-based open tasks.
    """
    task_templates = {
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
            "You make a embarrassing mistake in public. Write about your reaction.",
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
    }

    templates = task_templates.get(target_trait.lower(), task_templates.get("extraversion", []))
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


def score_questionnaire_response(response: str, task: Dict) -> float:
    """Score a questionnaire response using rule-based extraction."""
    raw = extract_questionnaire_score(
        response,
        scale=task.get("scale", 5),
        reverse=task.get("reverse", False),
    )
    if raw is not None:
        return raw
    return 0.5  # default neutral score if parsing fails


def run_irote_optimization(
    args,
    target_trait: str,
    evaluator: Evaluator,
    router: ModelRouter,
):
    """
    Full IROTE optimization loop (Algorithm 1).
    """
    output_dir = os.path.join(args.output_dir, target_trait)
    os.makedirs(output_dir, exist_ok=True)

    # Load questionnaires
    q_path = os.path.join(os.path.dirname(__file__), "data", "questionnaires.json")
    with open(q_path, "r") as f:
        questionnaires = {q["name"]: q for q in json.load(f)}

    # Get trait description
    trait_descriptions = {
        "extraversion": "Extraversion reflects being energetic, talkative, assertive, socially engaged, enthusiastic, and comfortable initiating interaction.",
        "agreeableness": "Agreeableness reflects being compassionate, cooperative, trusting, helpful, forgiving, and considerate towards others.",
        "conscientiousness": "Conscientiousness reflects being organized, disciplined, reliable, thorough, ambitious, and goal-directed.",
        "neuroticism": "Neuroticism reflects tendencies towards anxiety, emotional instability, worry, moodiness, and vulnerability to stress.",
        "openness": "Openness reflects being curious, creative, imaginative, open to new experiences, appreciative of art, and intellectually flexible.",
        "security": "Security reflects valuing safety, harmony, stability of society, relationships, and self.",
        "benevolence": "Benevolence reflects preserving and enhancing the welfare of those with whom one is in frequent personal contact.",
        "universalism": "Universalism reflects understanding, appreciation, tolerance, and protection for the welfare of all people and for nature.",
        "care": "Care/Harm reflects cherishing and protecting others, and empathizing with those who suffer.",
        "fairness": "Fairness/Cheating reflects rendering justice according to shared rules, and avoiding cheating.",
    }
    trait_desc = trait_descriptions.get(target_trait.lower(), f"The trait of {target_trait}")

    print(f"\n{'='*70}")
    print(f"IROTE Optimization: {target_trait}")
    print(f"Model: {args.model_name}")
    print(f"Iterations: {args.max_iteration}, K={args.init_reflection_num}, M1={args.M1}, M2={args.M2}")
    print(f"{'='*70}")

    # Load tasks
    questionnaire_tasks = load_optimize_tasks(args.evaluation_system, target_trait, questionnaires)
    open_tasks = load_open_tasks(target_trait, n_tasks=5)
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

    if len(trait_reflections) < args.init_reflection_num:
        print(f"Warning: Only {len(trait_reflections)} initial reflections available (need {args.init_reflection_num})")
        print("Generating additional reflections via LLM...")
        from irote_core import build_behavior_sampling_prompt, extract_json_list
        # Generate more via LLM
        gen_prompt = f"""Generate {args.init_reflection_num} diverse first-person self-reflections for the trait: {target_trait}.
Trait description: {trait_desc}

Each reflection should:
1. Describe self-perceived experiences, habits, values, or motivations
2. Use format: "I [verb]..., e.g.: [concrete example]"
3. Be under 50 words
4. NOT say "I am {target_trait}" directly
5. NOT include demographic info (age, gender, country, religion)

Output a JSON list of strings only."""
        messages = [[{"role": "user", "content": gen_prompt}]]
        responses = router.request_llm(
            conversations=messages,
            model=args.model_name,
            max_length=2048,
            temperature=0.8,
        )
        generated = extract_json_list(responses[0]) if responses else []
        trait_reflections.extend(generated)

    # Take first K reflections
    candidate_texts = trait_reflections[:args.init_reflection_num]
    print(f"Initial reflections: {len(candidate_texts)}")

    # Convert task prompts to strings for irote_core
    task_prompts = [t["prompt"] for t in all_tasks]

    # Initialize tracking
    best_reflection = None
    best_score = -float("inf")
    history = []

    # Main optimization loop (Algorithm 1)
    for t in range(1, args.max_iteration + 1):
        print(f"\n{'='*70}")
        print(f"Iteration {t}/{args.max_iteration}")
        print(f"{'='*70}")

        current_reflection = best_reflection if best_reflection else "\n".join(
            [f"{i+1}. {r}" for i, r in enumerate(candidate_texts[:5])]
        )

        # Run one IROTE iteration
        result = run_irote_iteration(
            router=router,
            iteration=t,
            current_reflection=current_reflection,
            candidate_reflections=candidate_texts,
            optimize_tasks=task_prompts,
            trait_name=target_trait,
            trait_description=trait_desc,
            model_name=args.model_name,
            judge_model=args.model_name,  # use same model as judge
            K=args.init_reflection_num,
            M1=args.M1,
            M2=args.M2,
            max_words=args.words_limit,
            beta=args.beta,
            gamma=0.3,
            alpha_len=0.1,
            alpha_dup=0.1,
        )

        # Update best reflection
        new_best = result["best_reflection"]
        new_best_score = new_best.scores.get("final_score", result.get("r2_score", 0))
        if new_best_score > best_score:
            best_score = new_best_score
            best_reflection = new_best.text

        # Update candidates for next iteration
        if result["ranked_candidates"]:
            candidate_texts = [r.text for r in result["ranked_candidates"][:args.init_reflection_num]]

        # Save iteration results
        iter_save = {
            "iteration": t,
            "current_reflection": current_reflection,
            "compacted_reflection": result["compacted_reflection"],
            "r2_score": result["r2_score"],
            "avg_trait_score": result["avg_trait_score"],
            "best_reflection": new_best.text,
            "best_score": new_best.scores,
            "n_candidates": len(result["ranked_candidates"]),
        }
        history.append(iter_save)

        with open(os.path.join(output_dir, f"iter_{t}.json"), "w") as f:
            json.dump(iter_save, f, indent=2, ensure_ascii=False)

        print(f"\nIteration {t} Summary:")
        print(f"  R2 score: {result['r2_score']:.4f}")
        print(f"  Avg trait score: {result['avg_trait_score']:.4f}")
        print(f"  Best candidate score: {new_best.scores.get('final_score', 0):.4f}")
        print(f"  Global best score: {best_score:.4f}")

        # Early stopping check
        if t >= 3 and len(history) >= 2:
            prev_score = history[-2]["best_score"].get("final_score", 0)
            curr_score = history[-1]["best_score"].get("final_score", 0)
            if curr_score - prev_score < 0.01:
                print(f"\nEarly stopping: score improvement < 0.01 for 2 consecutive iterations")
                break

    # Save final result
    final_result = {
        "trait": target_trait,
        "model": args.model_name,
        "iterations": len(history),
        "best_reflection": best_reflection,
        "best_score": best_score,
        "history": history,
    }
    with open(os.path.join(output_dir, "final_reflection.json"), "w") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Optimization Complete!")
    print(f"Best reflection ({best_score:.4f}):")
    print(f"{best_reflection}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")

    return final_result


def main():
    parser = argparse.ArgumentParser(description="IROTE Full Optimization")

    # Model settings
    parser.add_argument("--model_name", type=str, default="MiMo-v2.5-Pro")
    parser.add_argument("--eval_model_name", type=str, default=None,
                        help="Judge model (defaults to model_name)")

    # Trait settings
    parser.add_argument("--evaluation_system", type=str, default="personality",
                        choices=["value", "personality", "moral"])
    parser.add_argument("--specific_traits", type=str, default="extraversion",
                        help="Comma-separated trait names")

    # IROTE hyperparameters (matching paper)
    parser.add_argument("--max_iteration", type=int, default=5)
    parser.add_argument("--init_reflection_num", type=int, default=10, help="K in paper")
    parser.add_argument("--M1", type=int, default=3, help="Behaviors per candidate")
    parser.add_argument("--M2", type=int, default=6, help="Responses per task")
    parser.add_argument("--beta", type=float, default=1.0, help="Evocativeness weight")
    parser.add_argument("--words_limit", type=int, default=50, help="Max reflection words")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/irote_results")

    args = parser.parse_args()
    if args.eval_model_name is None:
        args.eval_model_name = args.model_name

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize router
    print(f"Initializing model: {args.model_name}")
    router = ModelRouter(
        model_names=[args.model_name, args.eval_model_name],
        temperature=1.0,
        top_p=0.95,
        max_model_len=1024,
    )

    # Initialize evaluator
    evaluator = Evaluator(
        router=router,
        evaluation_system=args.evaluation_system,
        tasks=["survey"],
    )

    # Run optimization for each trait
    traits = [t.strip() for t in args.specific_traits.split(",")]
    for trait in traits:
        run_irote_optimization(args, trait, evaluator, router)


if __name__ == "__main__":
    main()
