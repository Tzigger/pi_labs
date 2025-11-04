"""Benchmark script for comparing Huang's median filter with standard median filter."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .filters import load_image
from .noise import apply_huang_median_filter, apply_median_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Huang vs standard median filter performance"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("Images/Fig5.03_zgomotSarePiper.jpg"),
        help="Image to process (default: salt-and-pepper example)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[3, 5, 7, 9],
        help="Window sizes to test (default: 3 5 7 9)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per test (default: 3)",
    )
    parser.add_argument(
        "--wrap-edges",
        action="store_true",
        help="Use wrap-around padding",
    )
    return parser.parse_args()


def benchmark_filter(
    image: np.ndarray,
    filter_func: callable,
    size: int,
    wrap: bool,
    runs: int = 3,
) -> Tuple[float, float]:
    """
    Benchmark a filter function.

    Args:
        image: Input image
        filter_func: Filter function to benchmark
        size: Window size
        wrap: Whether to wrap edges
        runs: Number of runs

    Returns:
        Tuple of (mean_time, std_time) in seconds
    """
    times: List[float] = []

    # Warmup run
    _ = filter_func(image, size=size, wrap=wrap)

    for _ in range(runs):
        start = time.perf_counter()
        _ = filter_func(image, size=size, wrap=wrap)
        end = time.perf_counter()
        times.append(end - start)

    return float(np.mean(times)), float(np.std(times))


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1_000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image '{args.image}' does not exist.")

    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    print(f"Number of runs per test: {args.runs}")
    print(f"Edge handling: {'wrap-around' if args.wrap_edges else 'zero-padding'}")
    print()

    results: Dict[int, Dict[str, Tuple[float, float]]] = {}

    for size in args.sizes:
        print(f"Testing window size {size}x{size}...")
        results[size] = {}

        # Benchmark standard median filter
        print(f"  Running standard median filter...")
        mean_time, std_time = benchmark_filter(
            image, apply_median_filter, size, args.wrap_edges, args.runs
        )
        results[size]["median"] = (mean_time, std_time)
        print(f"    Time: {format_time(mean_time)} ± {format_time(std_time)}")

        # Benchmark Huang median filter
        print(f"  Running Huang median filter...")
        mean_time, std_time = benchmark_filter(
            image, apply_huang_median_filter, size, args.wrap_edges, args.runs
        )
        results[size]["huang"] = (mean_time, std_time)
        print(f"    Time: {format_time(mean_time)} ± {format_time(std_time)}")

        # Calculate speedup
        median_time = results[size]["median"][0]
        huang_time = results[size]["huang"][0]
        speedup = median_time / huang_time
        print(f"  Speedup: {speedup:.2f}x")
        print()

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Size':<8} {'Standard Median':<25} {'Huang Median':<25} {'Speedup':<10}")
    print("-" * 80)

    for size in args.sizes:
        median_mean, median_std = results[size]["median"]
        huang_mean, huang_std = results[size]["huang"]
        speedup = median_mean / huang_mean

        median_str = f"{format_time(median_mean)} ± {format_time(median_std)}"
        huang_str = f"{format_time(huang_mean)} ± {format_time(huang_std)}"
        speedup_str = f"{speedup:.2f}x"

        print(f"{size}x{size:<5} {median_str:<25} {huang_str:<25} {speedup_str:<10}")

    print("=" * 80)


if __name__ == "__main__":
    main()
