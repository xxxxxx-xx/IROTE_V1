"""
IROTE Optimization - Paper-Aligned Implementation
==================================================
Fixes applied:
1. [FIX #1] logprobs support via router.request_llm_with_logprobs
2. [FIX #2] PMI-based compactness via ProbabilityEstimator
3. [FIX #3] Connected evaluation pipeline (survey_handler)
4. [FIX #4] Retriever integrated for dedup
5. [FIX #5] Separate target_model and judge_model
6. [FIX #6] Complete trait templates (including MFT-Sanctity)
7. [FIX #7] Fixed questionnaire scoring
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
    OPEN_TASK_TEMPLATES,
    extract_questionnaire_score,
    ProbabilityEstimator,
)
from tools import Retriever
from eval_utils import Evaluator
from eval_utils.controller import Controller


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
        scale = q_data["scale"]
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


def evaluate_with_survey_handler(
    reflection: str,
    evaluator: Evaluator,
    target_trait: str,
    model_name: str,
    router: ModelRouter,
) -> Dict:
    """
    [FIX #3] Evaluate reflection using survey_handler (paper's evaluation pipeline).
    Returns questionnaire-based trait scores.
    """
    # Build the reflection prefix
    reflection_prefix = Controller.get_multi_shot_instruction(reflection)

    # Get handlers for evaluation
    handlers = evaluator.get_handlers()
    if not handlers:
        return {"error": "No evaluation handlers available"}

    results = {}
    for handler in handlers:
        try:
            eval_result = handler.get_target_results(
                target_trait=target_trait,
                model_name=model_name,
                shot_str=reflection_prefix,
                max_tokens=1024,
                temperature=0.01,
            )
            results.update(eval_result)
        except Exception as e:
            results[f"error_{handler.__class__.__name__}"] = str(e)

    return results


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

    # [FIX #4] Initialize Retriever for dedup
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

    # Initialize Retriever with trait reflections
    retriever = Retriever(
        model_name="dummy",  # Will use TF-IDF if SimCSE not available
        corpus_texts=trait_reflections,
        use_gpu=False,
    )

    candidate_texts = trait_reflections[:args.K]
    print(f"Initial reflections: {len(candidate_texts)}")

    # Task prompts
    task_prompts = [t["prompt"] for t in all_tasks]

    # Initialize ProbabilityEstimator for PMI-based compactness
    prob_estimator = ProbabilityEstimator()

    # [FIX #3] Initialize evaluator for survey-based evaluation
    evaluator = None
    try:
        evaluator = Evaluator(
            router=router,
            evaluation_system=args.evaluation_system,
            tasks=["survey"],
        )
        print(f"Evaluator initialized for {args.evaluation_system}")
    except Exception as e:
        print(f"Warning: Could not initialize evaluator: {e}")

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
            prob_estimator=prob_estimator,
        )

        # Update best
        new_best = result["best_reflection"]
        new_best_score = new_best.scores.get("final_score", 0)
        if new_best_score > best_score:
            best_score = new_best_score
            best_reflection = new_best.text

        # [FIX #4] Update Retriever with new candidates (dedup)
        if result["ranked_candidates"]:
            new_texts = [r.text for r in result["ranked_candidates"][:args.K]]
            retriever.dedup_and_add_corpus(new_texts, threshold=args.dedup_threshold)
            candidate_texts = new_texts

        # [FIX #3] Evaluate with survey_handler
        survey_scores = {}
        if evaluator and t % 2 == 0:  # Evaluate every 2 iterations
            print(f"\n[Eval] Running survey evaluation...")
            survey_scores = evaluate_with_survey_handler(
                reflection=new_best.text,
                evaluator=evaluator,
                target_trait=target_trait,
                model_name=args.target_model,
                router=router,
            )
            print(f"  Survey scores: {survey_scores}")

        # Save iteration
        iter_save = {
            "iteration": t,
            "current_reflection": current_reflection,
            "compacted_reflection": result["compacted_reflection"],
            "r2_score": result["r2_score"],
            "avg_q_score": result["avg_q_score"],
            "best_reflection": new_best.text,
            "best_score": new_best.scores,
            "survey_scores": survey_scores,
            "n_candidates": len(result["ranked_candidates"]),
            "corpus_size": len(retriever.corpus_texts),
        }
        history.append(iter_save)

        with open(os.path.join(output_dir, f"iter_{t}.json"), "w") as f:
            json.dump(iter_save, f, indent=2, ensure_ascii=False)

        print(f"\nIteration {t} Summary:")
        print(f"  R2: {result['r2_score']:.4f}, Avg q_ω: {result['avg_q_score']:.4f}")
        print(f"  Best score: {new_best_score:.4f}, Global best: {best_score:.4f}")
        print(f"  Corpus size: {len(retriever.corpus_texts)}")

        # Early stopping
        if t >= 3 and len(history) >= 2:
            prev = history[-2]["best_score"].get("final_score", 0)
            curr = history[-1]["best_score"].get("final_score", 0)
            if curr - prev < 0.01:
                print(f"\nEarly stopping: improvement < 0.01")
                break

    # [FIX #3] Final evaluation with survey_handler
    final_survey_scores = {}
    if evaluator:
        print(f"\n[Final Eval] Running final survey evaluation...")
        final_survey_scores = evaluate_with_survey_handler(
            reflection=best_reflection,
            evaluator=evaluator,
            target_trait=target_trait,
            model_name=args.target_model,
            router=router,
        )
        print(f"  Final survey scores: {final_survey_scores}")

    # Save final result
    final = {
        "trait": target_trait,
        "target_model": args.target_model,
        "judge_model": args.judge_model,
        "params": {"K": args.K, "M1": args.M1, "M2": args.M2, "beta": args.beta, "T": args.max_iteration},
        "iterations": len(history),
        "best_reflection": best_reflection,
        "best_score": best_score,
        "final_survey_scores": final_survey_scores,
        "history": history,
    }
    with open(os.path.join(output_dir, "final_reflection.json"), "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Optimization Complete!")
    print(f"Best reflection (score={best_score:.4f}):")
    print(f"{best_reflection}")
    if final_survey_scores:
        print(f"Final survey scores: {final_survey_scores}")
    print(f"Results: {output_dir}")
    print(f"{'='*70}")

    return final


def main():
    parser = argparse.ArgumentParser(description="IROTE Optimization")

    # Model settings
    parser.add_argument("--target_model", type=str, default="MiMo-v2.5-Pro",
                        help="Model being optimized")
    parser.add_argument("--judge_model", type=str, default="MiMo-v2.5-Pro",
                        help="Model evaluating responses (separate to reduce bias)")

    # Trait
    parser.add_argument("--evaluation_system", type=str, default="personality",
                        choices=["value", "personality", "moral"])
    parser.add_argument("--specific_traits", type=str, default="extraversion")

    # IROTE hyperparameters
    parser.add_argument("--max_iteration", type=int, default=5)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--M1", type=int, default=3)
    parser.add_argument("--M2", type=int, default=6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--words_limit", type=int, default=50)
    parser.add_argument("--n_open_tasks", type=int, default=5)
    parser.add_argument("--dedup_threshold", type=float, default=0.8,
                        help="Similarity threshold for deduplication")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/irote_results")

    args = parser.parse_args()

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
