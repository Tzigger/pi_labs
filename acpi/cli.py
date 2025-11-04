"""Command-line interface for running the ACPI image processing workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from .filters import convolve_image, load_image, save_image
from .noise import (
    apply_gaussian_filter,
    apply_huang_median_filter,
    apply_mean_filter,
    apply_median_filter,
)
from .contours import (
    apply_prewitt,
    apply_sobel,
    normalize_gradient_for_display,
    zoom_bilinear,
    zoom_by_duplication_2x,
    zoom_by_linear_interpolation_2x,
    zoom_nearest_neighbor,
)

# Kernels come straight from the assignment specification.
PRESET_KERNELS: Dict[str, Tuple[np.ndarray, float]] = {
    "blur3": (
        np.array([[0.0, 0.2, 0.0], [0.2, 0.2, 0.2], [0.0, 0.2, 0.0]], dtype=np.float32),
        1.0,
    ),
    "blur5": (
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        ),
        1.0 / 13.0,
    ),
    "motion": (
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
        1.0 / 9.0,
    ),
}

BEST_FILTER_BY_NOISE = {
    "uniform": "mean",
    "gaussian": "gaussian",
    "sarepiper": "huang",
    "sare_piper": "huang",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACPI convolution and denoising toolkit"
    )
    parser.add_argument("--images-dir", type=Path, default=Path("Images"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--wrap-edges",
        action="store_true",
        help="Use wrap-around padding for convolutions",
    )
    
    # ========== Previous lab options (commented functionality) ==========
    parser.add_argument(
        "--skip-convolution",
        action="store_true",
        help="Skip the convolution demo stage",
    )
    parser.add_argument(
        "--skip-denoising", action="store_true", help="Skip the noise reduction stage"
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Window size for mean and median filters",
    )
    parser.add_argument(
        "--gaussian-size",
        type=int,
        default=5,
        help="Window size for the Gaussian filter",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.0,
        help="Standard deviation for the Gaussian filter",
    )
    
    # ========== New lab options - Edge Detection & Zooming ==========
    parser.add_argument(
        "--edge-detection",
        action="store_true",
        help="Enable edge detection (Sobel and Prewitt)",
    )
    parser.add_argument(
        "--zoom-2x",
        action="store_true",
        help="Enable 2x2 zooming (duplication and linear interpolation)",
    )
    parser.add_argument(
        "--zoom-arbitrary",
        action="store_true",
        help="Enable arbitrary zoom with specified scale factors",
    )
    parser.add_argument(
        "--scale-x",
        type=float,
        default=2.0,
        help="Horizontal scale factor for arbitrary zoom (default: 2.0)",
    )
    parser.add_argument(
        "--scale-y",
        type=float,
        default=2.0,
        help="Vertical scale factor for arbitrary zoom (default: 2.0)",
    )
    parser.add_argument(
        "--show-table",
        action="store_true",
        help="Show results in a table layout (Lab 6 contours and zoom)",
    )
    parser.add_argument(
        "--table-image",
        type=str,
        default="football.jpg",
        help="Image to use for table display (default: football.jpg)",
    )
    
    return parser.parse_args()


def iter_images(images_dir: Path) -> Iterable[Path]:
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
    )


def run_convolution_stage(
    image_paths: Iterable[Path], output_dir: Path, wrap: bool
) -> None:
    stage_dir = output_dir / "convolution"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name, (kernel, factor) in PRESET_KERNELS.items():
        kernel_dir = stage_dir / name
        kernel_dir.mkdir(exist_ok=True)
        for image_path in image_paths:
            src = load_image(image_path)
            result = convolve_image(src, kernel, factor=factor, wrap=wrap)
            save_path = kernel_dir / f"{image_path.stem}_{name}{image_path.suffix}"
            save_image(result, save_path)


def run_denoising_stage(
    image_paths: Iterable[Path],
    output_dir: Path,
    wrap: bool,
    kernel_size: int,
    gaussian_size: int,
    gaussian_sigma: float,
) -> None:
    stage_dir = output_dir / "denoise"
    stage_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        src = load_image(image_path)
        results: Dict[str, np.ndarray] = {}
        filters = {
            "mean": lambda img: apply_mean_filter(img, size=kernel_size, wrap=wrap),
            "gaussian": lambda img: apply_gaussian_filter(
                img, size=gaussian_size, sigma=gaussian_sigma, wrap=wrap
            ),
            "median": lambda img: apply_median_filter(img, size=kernel_size, wrap=wrap),
            "huang": lambda img: apply_huang_median_filter(
                img, size=kernel_size, wrap=wrap
            ),
        }

        for filter_name, func in filters.items():
            filtered = func(src)
            results[filter_name] = filtered
            filter_dir = stage_dir / filter_name
            filter_dir.mkdir(exist_ok=True)
            save_path = (
                filter_dir / f"{image_path.stem}_{filter_name}{image_path.suffix}"
            )
            save_image(filtered, save_path)

        recommendation = pick_best_filter(image_path.stem)
        if recommendation and recommendation in results:
            best_dir = stage_dir / "best"
            best_dir.mkdir(exist_ok=True)
            save_path = (
                best_dir / f"{image_path.stem}_{recommendation}{image_path.suffix}"
            )
            save_image(results[recommendation], save_path)


def pick_best_filter(stem: str) -> str | None:
    token = stem.lower()
    for key, value in BEST_FILTER_BY_NOISE.items():
        if key in token:
            return value
    return None


def run_edge_detection_stage(
    image_paths: Iterable[Path], output_dir: Path, wrap: bool
) -> None:
    """Run edge detection using Sobel and Prewitt filters.
    
    For each image, computes:
    - Sobel Gx (horizontal gradient)
    - Sobel Gy (vertical gradient)
    - Sobel magnitude (combined)
    - Prewitt Gx (horizontal gradient)
    - Prewitt Gy (vertical gradient)
    - Prewitt magnitude (combined)
    """
    stage_dir = output_dir / "edge_detection"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path in image_paths:
        src = load_image(image_path)
        
        # ========== Sobel Edge Detection ==========
        sobel_dir = stage_dir / "sobel"
        sobel_dir.mkdir(exist_ok=True)
        
        gx_sobel, gy_sobel, mag_sobel = apply_sobel(src, wrap=wrap)
        
        # Normalize gradients for display
        gx_sobel_norm = normalize_gradient_for_display(gx_sobel)
        gy_sobel_norm = normalize_gradient_for_display(gy_sobel)
        
        # Save Sobel results
        save_image(gx_sobel_norm, sobel_dir / f"{image_path.stem}_sobel_gx{image_path.suffix}")
        save_image(gy_sobel_norm, sobel_dir / f"{image_path.stem}_sobel_gy{image_path.suffix}")
        save_image(mag_sobel, sobel_dir / f"{image_path.stem}_sobel_magnitude{image_path.suffix}")
        
        # ========== Prewitt Edge Detection ==========
        prewitt_dir = stage_dir / "prewitt"
        prewitt_dir.mkdir(exist_ok=True)
        
        gx_prewitt, gy_prewitt, mag_prewitt = apply_prewitt(src, wrap=wrap)
        
        # Normalize gradients for display
        gx_prewitt_norm = normalize_gradient_for_display(gx_prewitt)
        gy_prewitt_norm = normalize_gradient_for_display(gy_prewitt)
        
        # Save Prewitt results
        save_image(gx_prewitt_norm, prewitt_dir / f"{image_path.stem}_prewitt_gx{image_path.suffix}")
        save_image(gy_prewitt_norm, prewitt_dir / f"{image_path.stem}_prewitt_gy{image_path.suffix}")
        save_image(mag_prewitt, prewitt_dir / f"{image_path.stem}_prewitt_magnitude{image_path.suffix}")


def run_zoom_2x_stage(image_paths: Iterable[Path], output_dir: Path) -> None:
    """Run 2x2 zooming using duplication and linear interpolation.
    
    For each image, creates:
    - Zoomed by duplication (4x size)
    - Zoomed by linear interpolation (4x size)
    """
    stage_dir = output_dir / "zoom_2x"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path in image_paths:
        src = load_image(image_path)
        
        # ========== Zoom by Duplication ==========
        dup_dir = stage_dir / "duplication"
        dup_dir.mkdir(exist_ok=True)
        
        zoomed_dup = zoom_by_duplication_2x(src)
        save_image(zoomed_dup, dup_dir / f"{image_path.stem}_zoom_dup_2x{image_path.suffix}")
        
        # ========== Zoom by Linear Interpolation ==========
        interp_dir = stage_dir / "linear_interpolation"
        interp_dir.mkdir(exist_ok=True)
        
        zoomed_interp = zoom_by_linear_interpolation_2x(src)
        save_image(zoomed_interp, interp_dir / f"{image_path.stem}_zoom_interp_2x{image_path.suffix}")


def run_zoom_arbitrary_stage(
    image_paths: Iterable[Path], output_dir: Path, scale_x: float, scale_y: float
) -> None:
    """Run arbitrary zooming using nearest neighbor and bilinear interpolation.
    
    For each image, creates:
    - Zoomed by nearest neighbor
    - Zoomed by bilinear interpolation
    """
    stage_dir = output_dir / "zoom_arbitrary"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    scale_str = f"{scale_x}x{scale_y}".replace(".", "_")
    
    for image_path in image_paths:
        src = load_image(image_path)
        
        # ========== Nearest Neighbor ==========
        nn_dir = stage_dir / "nearest_neighbor"
        nn_dir.mkdir(exist_ok=True)
        
        zoomed_nn = zoom_nearest_neighbor(src, scale_x, scale_y)
        save_image(zoomed_nn, nn_dir / f"{image_path.stem}_zoom_nn_{scale_str}{image_path.suffix}")
        
        # ========== Bilinear Interpolation ==========
        bilinear_dir = stage_dir / "bilinear"
        bilinear_dir.mkdir(exist_ok=True)
        
        zoomed_bilinear = zoom_bilinear(src, scale_x, scale_y)
        save_image(zoomed_bilinear, bilinear_dir / f"{image_path.stem}_zoom_bilinear_{scale_str}{image_path.suffix}")


def main() -> None:
    args = parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory '{args.images_dir}' does not exist.")

    image_paths = list(iter_images(args.images_dir))
    if not image_paths:
        raise FileNotFoundError(f"No images found in '{args.images_dir}'.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # LABORATOR ANTERIOR - Convoluție și Denoising
    # Procesările de mai jos au fost comentate pentru a trece la laboratorul
    # despre contururi. Funcțiile rămân disponibile și pot fi reactivate.
    # ============================================================================
    
    # if not args.skip_convolution:
    #     run_convolution_stage(image_paths, args.output_dir, args.wrap_edges)

    # if not args.skip_denoising:
    #     run_denoising_stage(
    #         image_paths,
    #         args.output_dir,
    #         args.wrap_edges,
    #         args.kernel_size,
    #         args.gaussian_size,
    #         args.gaussian_sigma,
    #     )
    
    # ============================================================================
    # LABORATOR NOU - Detectare Contururi și Zoom
    # ============================================================================
    
    # Show table if requested (Lab 6 style)
    if args.show_table:
        from .contours_table import prepare_images, show_table
        
        print(f"Preparing table view with image: {args.table_image}")
        print(f"Scale factors: {args.scale_x}x{args.scale_y}")
        
        # Create a simple namespace for table preparation
        table_args = argparse.Namespace(
            images_dir=args.images_dir,
            image=args.table_image,
            scale_x=args.scale_x,
            scale_y=args.scale_y,
            wrap_edges=args.wrap_edges,
        )
        
        images = prepare_images(table_args)
        show_table(images, args.scale_x, args.scale_y)
        return  # Exit after showing table
    
    # Otherwise, run individual operations and save to disk
    if args.edge_detection:
        print("Running edge detection (Sobel and Prewitt)...")
        run_edge_detection_stage(image_paths, args.output_dir, args.wrap_edges)
        print(f"  ✓ Edge detection results saved to {args.output_dir / 'edge_detection'}")
    
    if args.zoom_2x:
        print("Running 2x2 zoom (duplication and linear interpolation)...")
        run_zoom_2x_stage(image_paths, args.output_dir)
        print(f"  ✓ 2x2 zoom results saved to {args.output_dir / 'zoom_2x'}")
    
    if args.zoom_arbitrary:
        print(f"Running arbitrary zoom ({args.scale_x}x{args.scale_y})...")
        run_zoom_arbitrary_stage(image_paths, args.output_dir, args.scale_x, args.scale_y)
        print(f"  ✓ Arbitrary zoom results saved to {args.output_dir / 'zoom_arbitrary'}")
    
    # Show usage hint if no operations were selected
    if not (args.edge_detection or args.zoom_2x or args.zoom_arbitrary):
        print("\n⚠️  No operations selected!")
        print("Use one or more of the following flags:")
        print("  --show-table        : Display results in a table (recommended for Lab 6)")
        print("  --edge-detection    : Detect edges using Sobel and Prewitt filters")
        print("  --zoom-2x           : Zoom images 2x2 using duplication and interpolation")
        print("  --zoom-arbitrary    : Zoom images with custom scale factors (--scale-x, --scale-y)")
        print("\nExample: python -m acpi.cli --show-table")
        print("Example: python -m acpi.cli --edge-detection --zoom-2x")


if __name__ == "__main__":
    main()
