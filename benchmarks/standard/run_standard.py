#!/usr/bin/env python3
"""
Run all standard benchmarks.

Usage:
    python benchmarks/standard/run_standard.py               # Full run
    python benchmarks/standard/run_standard.py --smoke-test   # Quick validation (10 examples)
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Run DimensionalBase standard benchmarks")
    parser.add_argument("--smoke-test", action="store_true", help="Quick run with 10 examples")
    parser.add_argument("--output-dir", default="benchmarks/standard/", help="Output directory")
    args = parser.parse_args()

    max_q = 10 if args.smoke_test else None
    budgets = [500, 2000] if args.smoke_test else [500, 1000, 2000, 4000]

    print("=" * 60)
    print("DimensionalBase Standard Benchmark Suite")
    print("=" * 60)

    results = {}

    # --- LongBench v2 ---
    print("\n--- LongBench v2 ---")
    try:
        from benchmarks.standard.longbench_v2 import LongBenchV2Evaluator

        evaluator = LongBenchV2Evaluator(max_questions=max_q, budgets=budgets)
        evaluator.evaluate(method_name="dimensionalbase")
        evaluator.save_results(f"{args.output_dir}/results_longbench_v2.json")
        results["longbench_v2"] = evaluator.summary()
        print("  LongBench v2: COMPLETE")
        for budget, metrics in evaluator.summary().items():
            print(f"    Budget {budget}: F1={metrics['avg_f1']:.3f}, "
                  f"Efficiency={metrics['avg_token_efficiency']:.3f}")
    except ImportError as e:
        print(f"  LongBench v2: SKIPPED ({e})")
        results["longbench_v2"] = {"error": str(e)}
    except Exception as e:
        print(f"  LongBench v2: FAILED ({e})")
        results["longbench_v2"] = {"error": str(e)}

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary_path = f"{args.output_dir}/benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
