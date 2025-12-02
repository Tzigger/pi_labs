# ACPI Image Processing Toolkit

This project implements convolution-based filters, noise reduction utilities, edge detection, and image zooming operations in Python for the ACPI laboratory assignments. Place your source images inside the `Images/` directory, then run the CLI to generate processed versions under `output/`.

## Features

### Laboratories 1-5: Convolution and Noise Reduction
- Generic 2D convolution with configurable kernels, scaling factor, bias, and edge handling.
- Preconfigured smoothing filters (blur and motion blur) as specified in the assignment.
- Noise reduction using arithmetic mean, Gaussian, median, and **Huang's optimized median** filters.
- **Huang's algorithm**: Histogram-based median filter with significant performance improvements for larger window sizes.
- Batch processing of all images located in `Images/`.
- Laboratory exercises suite for comprehensive filter comparison.
- Performance benchmarking tools for filter comparison.

### Laboratory 6: Edge Detection and Image Zooming
- **Edge Detection**: Sobel and Prewitt gradient operators for detecting edges
  - Individual horizontal (Gx) and vertical (Gy) gradients
  - Combined gradient magnitude
- **Image Zooming (2x2)**:
  - Duplication method (pixel replication)
  - Linear interpolation (bilinear smoothing)
- **Image Zooming (Arbitrary Scale)**:
  - Nearest neighbor interpolation
  - Bilinear interpolation
- **Table View**: Display all results in a comprehensive table layout

## Getting Started

1. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place the provided images in `Images/` (examples already included)

3. **For Laboratory 6 (Edge Detection & Zooming)** - View results in table:

```bash
python -m acpi.cli --show-table
```

This displays all edge detection and zooming operations in a comprehensive table view.

4. **For previous laboratories** - Run the processing script:

```bash
python -m acpi.cli --wrap-edges
```

Use `--wrap-edges` to apply toroidal wrapping during convolution when experimenting with the alternative boundary condition. By default, zero-padding is used.

Generated outputs will be organized in subfolders inside `output/`, grouped by the filter applied.

To preview filters without writing files, use the viewer utility. It opens a window that shows the original image next to the filtered result:

```bash
python -m acpi.viewer --filters median mean gaussian
```

By default the viewer loads `Images/Fig5.03_zgomotSarePiper.jpg`. Pass `--image` if you want to inspect another file.

For a summary figure laid out like a table (original vs blurred football on the first row, noise reduction on the remaining rows), run:

```bash
python -m acpi.table
```

This command displays the composite without saving intermediary files. Use `--wrap-edges` or the size arguments to experiment with the window sizes.

## CLI reference

### Laboratory 6: Edge Detection and Zooming

**Show results in table (recommended)**:
```bash
python -m acpi.cli --show-table
```

**Options for table display**:
- `--table-image FILENAME`: Image to process (default: football.jpg)
- `--scale-x FACTOR`: Horizontal zoom factor (default: 1.5)
- `--scale-y FACTOR`: Vertical zoom factor (default: 1.5)
- `--wrap-edges`: Use wrap-around padding

**Examples**:
```bash
# Show table with default settings
python -m acpi.cli --show-table

# Use a different image and zoom factor
python -m acpi.cli --show-table --table-image coins.png --scale-x 2.0 --scale-y 2.0

# Apply wrap-around padding
python -m acpi.cli --show-table --wrap-edges
```

**Save results to disk** (instead of table view):
```bash
# Edge detection (Sobel and Prewitt)
python -m acpi.cli --edge-detection

# 2x2 zooming (duplication and linear interpolation)
python -m acpi.cli --zoom-2x

# Arbitrary zooming with custom scale factors
python -m acpi.cli --zoom-arbitrary --scale-x 1.5 --scale-y 1.5

# Run all operations
python -m acpi.cli --edge-detection --zoom-2x --zoom-arbitrary
```

### Previous Laboratories: Convolution and Denoising

```
python -m acpi.cli [options]

Options:
  --images-dir PATH       Directory with source images (default: Images/)
  --output-dir PATH       Directory where results are saved (default: output/)
  --wrap-edges            Use wrap-around padding instead of zero-padding
  --skip-convolution      Skip the convolution demo stage
  --skip-denoising        Skip the noise reduction stage
  --kernel-size N         Window size for the arithmetic mean and median filters (default: 3)
  --gaussian-size N       Window size for the Gaussian filter (default: 5)
  --gaussian-sigma F      Standard deviation for the Gaussian filter (default: 1.0)
```

**Note**: The convolution and denoising operations are currently commented out in the CLI. To use them, uncomment the relevant sections in `acpi/cli.py`.

## Recommended filters for noisy images
- Uniform noise: arithmetic mean filter.
- Gaussian noise: Gaussian filter with `sigma=1.0` and window size `5`.
- Salt-and-pepper noise: **Huang's median filter** (faster) or standard median filter.

These defaults are already applied by the CLI when it detects the example filenames. The script also exports the results for each filter so you can compare them visually.

## Huang's Median Filter Algorithm

Huang's algorithm is an optimized approach to median filtering that significantly reduces computational complexity:

**Standard Median Filter Complexity**: O(n × m × w²) where n×m is the image size and w is the window size.

**Huang's Algorithm Complexity**: O(n × m × w) - achieves speedup by reusing histogram data.

### How It Works

1. **Row-by-row processing**: The algorithm processes each row from left to right.
2. **Histogram reuse**: For the first window in each row, it builds a complete histogram. For subsequent windows, it only updates the histogram by:
   - Removing the leftmost column that exits the window
   - Adding the rightmost column that enters the window
3. **Fast median finding**: Instead of sorting pixels, it finds the median by counting pixels in the histogram until reaching position N/2.

### Performance

For a 3×3 window, performance is similar to the standard approach. However, for larger windows (e.g., 7×7, 9×9), Huang's algorithm provides significant speedups (typically 2-5×) because it avoids repeatedly sorting large pixel arrays.

Use the benchmark script to measure performance on your system:

```bash
python -m acpi.benchmark --sizes 3 5 7 9 11
```

## Laboratory Exercises

Run the complete exercise suite to explore different filter configurations:

```bash
python -m acpi.exercises
```

This will generate results for:

1. **Exercise 1**: Mean filter with different window sizes (3×3, 5×5, 7×7, 9×9)
2. **Exercise 2**: Gaussian filter with different parameters (sizes 5×5, 7×7 and σ values 0.5, 1.0, 1.5, 2.0)
3. **Exercise 3**: Comparison between standard median and Huang's algorithm
4. **Exercise 4**: Filter effectiveness on different noise types
5. **Exercise 5**: Edge handling comparison (zero-padding vs wrap-around)

Run a specific exercise:

```bash
python -m acpi.exercises --exercise 3
```

### Benchmark Performance

Compare the performance of standard median vs Huang's algorithm:

```bash
python -m acpi.benchmark --image Images/Fig5.03_zgomotSarePiper.jpg --sizes 3 5 7 9 --runs 5
```

Options:
- `--image PATH`: Image to process
- `--sizes [N...]`: Window sizes to test
- `--runs N`: Number of runs per test for averaging
- `--wrap-edges`: Use wrap-around padding

### Interactive Viewer with Huang Filter

View filter results in real-time:

```bash
python -m acpi.viewer --filters huang median mean gaussian
```

This opens a window showing the original image next to the filtered results for comparison.
