#!/usr/bin/env python3
"""The in-run evaluation of one tier, written to evaluation_results.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .conformance import (
    collect_cluster_info,
    evaluate_clusters_hm,
    evaluate_clusters_im,
    run_single_model_baseline,
)
from .report import (
    compute_quality_metrics,
    compute_weighted_averages,
    print_headline_diagnostics,
    print_per_cluster_tables,
)


def export_results(
    input_dir: Path, tier: str,
    im_results: list[dict], weighted_avg: dict,
    hm_results: list[dict], weighted_avg_hm: dict,
    single_model: dict, global_hm: dict,
    quality_metrics: dict,
    coverage: dict, coverage_hm: dict,
    conformance: dict,
) -> None:
    """Write evaluation_results.json."""
    output = {
        "tier": tier,
        "conformance": conformance,
        "clusters": im_results,
        "weighted_avg": weighted_avg,
        "coverage": coverage,
        "clusters_hm": hm_results,
        "weighted_avg_hm": weighted_avg_hm,
        "coverage_hm": coverage_hm,
        "single_model": single_model,
        "global_hm": global_hm,
        "quality_metrics": quality_metrics,
    }
    results_path = input_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-tier evaluation")
    parser.add_argument("input_dir", help="Directory with cluster DFGs and traces")
    parser.add_argument("--tier", default="plaintext",
                        choices=["federated", "plaintext", "projected", "single_model"])
    # The pipeline passes both under --fast-eval; the conformance is token replay either way.
    parser.add_argument("--skip-log-baseline", action="store_true")
    parser.add_argument("--token-replay", action="store_true")
    args = parser.parse_args()
    # Which backend produced this tier's numbers; no other field of the file says.
    conformance = {
        "token_replay": bool(args.token_replay),
        "skip_log_baseline": bool(args.skip_log_baseline),
        "fast_eval": bool(args.token_replay and args.skip_log_baseline),
    }

    input_dir = Path(args.input_dir)

    dfg_files = sorted(input_dir.glob("cluster_*_dfg.json"))
    if not dfg_files:
        print(f"No cluster DFG files found in {input_dir}")
        sys.exit(1)
    trace_files = sorted(input_dir.glob("cluster_*_traces.json"))
    if not trace_files:
        print(f"No cluster trace files in {input_dir}")
        sys.exit(1)

    print(f"=== Tier: {args.tier} ===")
    print(f"Found {len(dfg_files)} cluster DFGs, {len(trace_files)} trace files\n")

    cluster_info = collect_cluster_info(input_dir, dfg_files)
    im_results, im_coverage = evaluate_clusters_im(input_dir, cluster_info)
    hm_results, hm_coverage = evaluate_clusters_hm(input_dir, cluster_info)

    wavg_im, wavg_hm = compute_weighted_averages(im_results, hm_results)
    single_model, global_hm = run_single_model_baseline(input_dir, dfg_files, trace_files)

    print_per_cluster_tables(
        im_results, hm_results,
        wavg_im, wavg_hm,
        single_model, global_hm,
    )
    quality_metrics = compute_quality_metrics(
        cluster_info, hm_results, wavg_hm, global_hm,
    )
    er_avg = (quality_metrics.get("entropic_relevance") or {}).get("weighted_avg")
    headline = print_headline_diagnostics(
        cluster_info, hm_results, wavg_hm,
        quality_metrics.get("entropy", {}),
        er_weighted_avg=er_avg,
    )
    quality_metrics["headline"] = headline

    export_results(
        input_dir, args.tier,
        im_results, wavg_im, hm_results, wavg_hm,
        single_model, global_hm,
        quality_metrics,
        im_coverage, hm_coverage,
        conformance,
    )


if __name__ == "__main__":
    main()
