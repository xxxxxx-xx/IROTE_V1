"""
IROTE Quick Start Script
========================
Unified entry point for running IROTE optimization with closed-source models.

Usage:
    # Basic usage with GPT-4o
    python run_irote.py --model_name GPT-4o --evaluation_system value

    # With a generic API model (configured in config.py)
    python run_irote.py --model_name DeepSeek-V3 --evaluation_system moral

    # Full options
    python run_irote.py \
        --model_name GPT-4o \
        --eval_model_name GPT-4o \
        --evaluation_system value \
        --max_iteration 5 \
        --words_limit 50 \
        --specific_traits "security,benevolence"
"""

import os
import sys
import argparse

# Ensure the codes directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PATH_TO_SIMCSE_MODEL


def check_prerequisites():
    """Check if all prerequisites are met before running."""
    errors = []

    # Check SimCSE model
    if not os.path.exists(PATH_TO_SIMCSE_MODEL):
        errors.append(
            f"[ERROR] SimCSE model not found at: {PATH_TO_SIMCSE_MODEL}\n"
            f"  Download from: https://huggingface.co/princeton-nlp/sup-simcse-roberta-large\n"
            f"  Then set IROTE_SIMCSE_PATH env var or edit config.py"
        )

    # Check reflection data
    data_dir = os.path.join(os.path.dirname(__file__), "data", "reflection_data")
    for fname in ["value.csv", "moral.csv", "personality.csv"]:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            errors.append(f"[ERROR] Missing reflection data: {fpath}")

    # Check questionnaires
    q_path = os.path.join(os.path.dirname(__file__), "data", "questionnaires.json")
    if not os.path.exists(q_path):
        errors.append(f"[ERROR] Missing questionnaires: {q_path}")

    if errors:
        print("=" * 60)
        print("Prerequisites check FAILED:")
        print("=" * 60)
        for e in errors:
            print(f"  {e}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("[OK] All prerequisites met.")


def main():
    parser = argparse.ArgumentParser(description="IROTE: In-Context Self-Reflective Optimization for Trait Elicitation")

    # Model settings
    parser.add_argument("--model_name", type=str, default="GPT-4o",
                        help="Target LLM name (must be registered in llm_interface.py or config.py)")
    parser.add_argument("--eval_model_name", type=str, default="GPT-4o",
                        help="LLM used for probability estimation in compactness optimization")

    # Evaluation settings
    parser.add_argument("--evaluation_system", type=str, default="value",
                        choices=["value", "personality", "moral"],
                        help="Trait system: value (STBHV), personality (BigFive), moral (MFT)")
    parser.add_argument("--tasks", type=str, default="survey",
                        help="Evaluation tasks (comma-separated)")

    # Optimization parameters (matching paper settings)
    parser.add_argument("--max_iteration", type=int, default=5,
                        help="Maximum optimization iterations (T in Algorithm 1)")
    parser.add_argument("--init_reflection_num", type=int, default=50,
                        help="Number of initial reflections per set (K in paper)")
    parser.add_argument("--num_item_shuffle", type=int, default=0,
                        help="Number of reflection order shuffles")
    parser.add_argument("--use_cot", type=bool, default=True,
                        help="Use Chain-of-Thought in optimization")
    parser.add_argument("--optimization_beam_size", type=int, default=3,
                        help="Beam size for optimization step")
    parser.add_argument("--num_k_shot", type=int, default=1,
                        help="Number of k-shot examples")
    parser.add_argument("--summarization_beam_size", type=int, default=2,
                        help="Beam size for summarization step")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Top-k reflections to keep per iteration")
    parser.add_argument("--words_limit", type=int, default=50,
                        help="Maximum words per reflection (50 in paper)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum tokens for generation")
    parser.add_argument("--avg_mode", type=str, default="min",
                        choices=["min", "max", "algorithmic_mean", "geometric_mean", "harmonic_mean"],
                        help="Score aggregation mode")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Similarity threshold for deduplication")
    parser.add_argument("--use_task_description", type=bool, default=True,
                        help="Include task description in prompts")

    # vLLM settings (unused for closed-source models, kept for compatibility)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    # Trait selection
    parser.add_argument("--specific_traits", type=str, default="",
                        help="Comma-separated list of specific traits to optimize (empty = all)")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/all_survey",
                        help="Output directory for results")

    args = parser.parse_args()

    # Check prerequisites
    check_prerequisites()

    # Import after path setup
    from models.router import ModelRouter
    from eval_utils import Evaluator
    from optimization import opt_main

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize model router
    print(f"\n{'='*60}")
    print(f"Initializing IROTE with model: {args.model_name}")
    print(f"Evaluation system: {args.evaluation_system}")
    print(f"{'='*60}\n")

    router = ModelRouter(
        model_names=[args.model_name, args.eval_model_name],
        model_dir=None,
        temperature=args.temperature,
        top_p=0.95,
        max_model_len=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )

    # Build evaluator
    evaluator = Evaluator(
        router=router,
        evaluation_system=args.evaluation_system,
        tasks=args.tasks.split(","),
    )
    trait_mapping = evaluator.get_trait_mapping()

    # Select target traits
    if args.specific_traits:
        specific_traits = [t.strip() for t in args.specific_traits.split(",")]
        for trait in specific_traits:
            assert trait in trait_mapping, f"{trait} is not in trait_mapping: {list(trait_mapping.keys())}"
        target_traits = specific_traits
    else:
        target_traits = list(trait_mapping.keys())

    print(f"Target traits: {target_traits}")
    print(f"Number of traits to optimize: {len(target_traits)}\n")

    # Run optimization for each trait
    for target_trait in target_traits:
        eval_save_dir = os.path.join(args.output_dir, f'{target_trait}_eval')
        evaluator = Evaluator(
            router=router,
            evaluation_system=args.evaluation_system,
            tasks=args.tasks.split(","),
            save_dir=eval_save_dir
        )
        opt_main(args, target_trait, evaluator, router)

    print(f"\n{'='*60}")
    print(f"Optimization complete! Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
