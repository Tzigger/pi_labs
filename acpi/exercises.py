"""Laboratory exercises for image filtering and processing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .filters import load_image, save_image
from .noise import (
    apply_gaussian_filter,
    apply_huang_median_filter,
    apply_mean_filter,
    apply_median_filter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Laboratory 4 exercises for image processing"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("Images"),
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/exercises"),
        help="Directory for output images",
    )
    parser.add_argument(
        "--exercise",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific exercise (default: run all)",
    )
    parser.add_argument(
        "--wrap-edges",
        action="store_true",
        help="Use wrap-around padding",
    )
    return parser.parse_args()


def exercise1_mean_filter_variants(
    image_path: Path, output_dir: Path, wrap: bool = False
) -> None:
    """
    Exercise 1: Apply mean filter with different window sizes.
    Test window sizes: 3x3, 5x5, 7x7, 9x9
    """
    print("\n" + "=" * 80)
    print("EXERCISE 1: Mean Filter with Different Window Sizes")
    print("=" * 80)

    if not image_path.exists():
        print(f"Warning: Image '{image_path}' not found. Skipping exercise 1.")
        return

    image = load_image(image_path)
    ex_dir = output_dir / "exercise1_mean_filter"
    ex_dir.mkdir(parents=True, exist_ok=True)

    sizes = [3, 5, 7, 9]
    for size in sizes:
        print(f"Applying mean filter with window size {size}x{size}...")
        filtered = apply_mean_filter(image, size=size, wrap=wrap)
        out_path = ex_dir / f"mean_{size}x{size}.png"
        save_image(filtered, out_path)
        print(f"  Saved: {out_path}")

    print("Exercise 1 completed!")


def exercise2_gaussian_filter_variants(
    image_path: Path, output_dir: Path, wrap: bool = False
) -> None:
    """
    Exercise 2: Apply Gaussian filter with different sigma values.
    Test configurations:
    - Size 5x5, sigma: 0.5, 1.0, 1.5, 2.0
    - Size 7x7, sigma: 0.5, 1.0, 1.5, 2.0
    """
    print("\n" + "=" * 80)
    print("EXERCISE 2: Gaussian Filter with Different Parameters")
    print("=" * 80)

    if not image_path.exists():
        print(f"Warning: Image '{image_path}' not found. Skipping exercise 2.")
        return

    image = load_image(image_path)
    ex_dir = output_dir / "exercise2_gaussian_filter"
    ex_dir.mkdir(parents=True, exist_ok=True)

    sizes = [5, 7]
    sigmas = [0.5, 1.0, 1.5, 2.0]

    for size in sizes:
        for sigma in sigmas:
            print(f"Applying Gaussian filter (size={size}x{size}, sigma={sigma})...")
            filtered = apply_gaussian_filter(image, size=size, sigma=sigma, wrap=wrap)
            out_path = ex_dir / f"gaussian_{size}x{size}_sigma{sigma:.1f}.png"
            save_image(filtered, out_path)
            print(f"  Saved: {out_path}")

    print("Exercise 2 completed!")


def exercise3_median_vs_huang(
    image_path: Path, output_dir: Path, wrap: bool = False
) -> None:
    """
    Exercise 3: Compare standard median filter with Huang's algorithm.
    Test window sizes: 3x3, 5x5, 7x7
    """
    print("\n" + "=" * 80)
    print("EXERCISE 3: Median Filter vs Huang's Algorithm")
    print("=" * 80)

    if not image_path.exists():
        print(f"Warning: Image '{image_path}' not found. Skipping exercise 3.")
        return

    image = load_image(image_path)
    ex_dir = output_dir / "exercise3_median_comparison"
    ex_dir.mkdir(parents=True, exist_ok=True)

    sizes = [3, 5, 7]

    for size in sizes:
        print(f"\nWindow size {size}x{size}:")

        # Standard median
        print(f"  Applying standard median filter...")
        median_filtered = apply_median_filter(image, size=size, wrap=wrap)
        median_path = ex_dir / f"median_standard_{size}x{size}.png"
        save_image(median_filtered, median_path)
        print(f"    Saved: {median_path}")

        # Huang median
        print(f"  Applying Huang median filter...")
        huang_filtered = apply_huang_median_filter(image, size=size, wrap=wrap)
        huang_path = ex_dir / f"median_huang_{size}x{size}.png"
        save_image(huang_filtered, huang_path)
        print(f"    Saved: {huang_path}")

        # Calculate difference
        diff = np.abs(median_filtered - huang_filtered)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  Difference: max={max_diff:.4f}, mean={mean_diff:.4f}")

        if max_diff > 0.1:
            diff_path = ex_dir / f"difference_{size}x{size}.png"
            # Amplify difference for visualization
            diff_amplified = np.clip(diff * 10, 0, 255)
            save_image(diff_amplified, diff_path)
            print(f"    Difference image saved: {diff_path}")

    print("\nExercise 3 completed!")


def exercise4_noise_comparison(
    images_dir: Path, output_dir: Path, wrap: bool = False
) -> None:
    """
    Exercise 4: Compare filter effectiveness on different noise types.
    Apply mean, Gaussian, median, and Huang filters to:
    - Uniform noise
    - Gaussian noise
    - Salt-and-pepper noise
    """
    print("\n" + "=" * 80)
    print("EXERCISE 4: Filter Effectiveness on Different Noise Types")
    print("=" * 80)

    ex_dir = output_dir / "exercise4_noise_comparison"
    ex_dir.mkdir(parents=True, exist_ok=True)

    noise_images = {
        "uniform": "Fig5.03_zgomotUniform.jpg",
        "gaussian": "Fig5.03_zgomotGaussian.jpg",
        "saltpepper": "Fig5.03_zgomotSarePiper.jpg",
    }

    filters = {
        "mean": lambda img: apply_mean_filter(img, size=3, wrap=wrap),
        "gaussian": lambda img: apply_gaussian_filter(
            img, size=5, sigma=1.0, wrap=wrap
        ),
        "median": lambda img: apply_median_filter(img, size=3, wrap=wrap),
        "huang": lambda img: apply_huang_median_filter(img, size=3, wrap=wrap),
    }

    for noise_type, filename in noise_images.items():
        image_path = images_dir / filename
        if not image_path.exists():
            print(f"Warning: Image '{image_path}' not found. Skipping {noise_type}.")
            continue

        print(f"\nProcessing {noise_type} noise image...")
        image = load_image(image_path)

        noise_dir = ex_dir / noise_type
        noise_dir.mkdir(exist_ok=True)

        # Save original
        orig_path = noise_dir / "original.png"
        save_image(image, orig_path)
        print(f"  Original saved: {orig_path}")

        # Apply all filters
        for filter_name, filter_func in filters.items():
            print(f"  Applying {filter_name} filter...")
            filtered = filter_func(image)
            out_path = noise_dir / f"{filter_name}_filtered.png"
            save_image(filtered, out_path)
            print(f"    Saved: {out_path}")

    print("\nExercise 4 completed!")


def exercise5_edge_handling_comparison(image_path: Path, output_dir: Path) -> None:
    """
    Exercise 5: Compare zero-padding vs wrap-around edge handling.
    Apply median filter with both methods.
    """
    print("\n" + "=" * 80)
    print("EXERCISE 5: Edge Handling Comparison")
    print("=" * 80)

    if not image_path.exists():
        print(f"Warning: Image '{image_path}' not found. Skipping exercise 5.")
        return

    image = load_image(image_path)
    ex_dir = output_dir / "exercise5_edge_handling"
    ex_dir.mkdir(parents=True, exist_ok=True)

    sizes = [3, 5, 7]

    for size in sizes:
        print(f"\nWindow size {size}x{size}:")

        # Zero-padding
        print(f"  Applying median filter with zero-padding...")
        zero_padded = apply_median_filter(image, size=size, wrap=False)
        zero_path = ex_dir / f"median_zeropad_{size}x{size}.png"
        save_image(zero_padded, zero_path)
        print(f"    Saved: {zero_path}")

        # Wrap-around
        print(f"  Applying median filter with wrap-around...")
        wrap_around = apply_median_filter(image, size=size, wrap=True)
        wrap_path = ex_dir / f"median_wraparound_{size}x{size}.png"
        save_image(wrap_around, wrap_path)
        print(f"    Saved: {wrap_path}")

        # Compare edges - extract border regions
        border_width = size
        # Top border
        top_zero = zero_padded[:border_width, :, :]
        top_wrap = wrap_around[:border_width, :, :]
        top_diff = np.abs(top_zero - top_wrap)

        # Left border
        left_zero = zero_padded[:, :border_width, :]
        left_wrap = wrap_around[:, :border_width, :]
        left_diff = np.abs(left_zero - left_wrap)

        max_diff = max(np.max(top_diff), np.max(left_diff))
        mean_diff = (np.mean(top_diff) + np.mean(left_diff)) / 2

        print(f"  Border difference: max={max_diff:.4f}, mean={mean_diff:.4f}")

    print("\nExercise 5 completed!")


def create_comparison_grid(
    images: Dict[str, np.ndarray], output_path: Path, grid_size: Tuple[int, int]
) -> None:
    """
    Create a comparison grid of images.

    Args:
        images: Dictionary mapping labels to images
        output_path: Path to save the grid
        grid_size: Tuple of (rows, cols)
    """
    import matplotlib.pyplot as plt

    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    labels = list(images.keys())

    for idx, (ax, label) in enumerate(zip(axes.flat, labels)):
        if label in images:
            img = np.clip(images[label], 0, 255).astype(np.uint8)
            ax.imshow(img)
            ax.set_title(label, fontsize=12)
        ax.axis("off")

    # Hide remaining axes
    for idx in range(len(labels), rows * cols):
        axes.flat[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison grid saved: {output_path}")


def main() -> None:
    args = parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory '{args.images_dir}' not found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("IMAGE PROCESSING LABORATORY - EXERCISE SET")
    print("=" * 80)
    print(f"Images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Edge handling: {'wrap-around' if args.wrap_edges else 'zero-padding'}")

    # Define test images
    uniform_noise = args.images_dir / "Fig5.03_zgomotUniform.jpg"
    gaussian_noise = args.images_dir / "Fig5.03_zgomotGaussian.jpg"
    saltpepper_noise = args.images_dir / "Fig5.03_zgomotSarePiper.jpg"

    # Run exercises
    if args.exercise is None or args.exercise == 1:
        exercise1_mean_filter_variants(uniform_noise, args.output_dir, args.wrap_edges)

    if args.exercise is None or args.exercise == 2:
        exercise2_gaussian_filter_variants(
            gaussian_noise, args.output_dir, args.wrap_edges
        )

    if args.exercise is None or args.exercise == 3:
        exercise3_median_vs_huang(saltpepper_noise, args.output_dir, args.wrap_edges)

    if args.exercise is None or args.exercise == 4:
        exercise4_noise_comparison(args.images_dir, args.output_dir, args.wrap_edges)

    if args.exercise is None or args.exercise == 5:
        exercise5_edge_handling_comparison(saltpepper_noise, args.output_dir)

    print("\n" + "=" * 80)
    print("ALL EXERCISES COMPLETED!")
    print("=" * 80)
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
