#!/usr/bin/env python3
"""
Visualization script for Swin-UNETR inference outputs.

Loads Zarr prediction files and generates 2D slice visualizations.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid function
import zarr
from scipy import ndimage


def load_zarr_prediction(prediction_path: Path) -> np.ndarray:
    """
    Load prediction from Zarr or NPY file.
    
    - Zarr: expected format CZYX or TCZYX (as saved by platform inference)
    - NPY:  expected format ZYXC (simple_inference.py output) or CZYX
    
    Returns: numpy array in CZYX format.
    """
    path = Path(prediction_path)
    
    # --- NPY branch (simple_inference.py output) ---
    if path.suffix == '.npy':
        arr = np.load(path)
        print(f"Loaded NPY prediction with shape: {arr.shape}")
        
        if arr.ndim == 4:
            # Heuristic: treat as ZYXC (Z, Y, X, C), move channels to front
            # simple_inference.py writes (Z, Y, X, 1)
            if arr.shape[-1] >= 1:
                # ZYXC -> CZYX
                arr = np.moveaxis(arr, -1, 0)  # (C, Z, Y, X)
            else:
                raise ValueError(f"Unexpected NPY shape (last dim=0): {arr.shape}")
        elif arr.ndim == 3:
            # (Z, Y, X) -> (1, Z, Y, X)
            arr = arr[None, ...]
        else:
            raise ValueError(
                f"Unexpected NPY array shape: {arr.shape}, expected 3D (Z,Y,X) or 4D (Z,Y,X,C)"
            )
        
        return arr
    
    # --- Zarr branch (original behaviour) ---
    zarr_group = zarr.open(str(path), mode='r')
    
    # Find the main array (usually named '0' or similar)
    if '0' in zarr_group:
        arr = zarr_group['0'][:]
    else:
        # Try to find any array in the group
        keys = list(zarr_group.keys())
        if keys:
            arr = zarr_group[keys[0]][:]
        else:
            raise ValueError(f"No array found in zarr file: {path}")
    
    # Handle TCZYX -> CZYX conversion if needed
    if arr.ndim == 5:  # TCZYX
        arr = arr[0]  # Take first timepoint -> CZYX
    elif arr.ndim == 4:  # CZYX
        pass  # Already correct
    else:
        raise ValueError(f"Unexpected array shape: {arr.shape}, expected CZYX or TCZYX")
    
    return arr


def apply_sigmoid(prediction: np.ndarray) -> np.ndarray:
    """
    Apply sigmoid activation to convert logits to probabilities.
    
    Args:
        prediction: CZYX array with logits
        
    Returns:
        CZYX array with probabilities [0, 1]
    """
    return expit(prediction)


def plot_intensity_histogram(
    prediction: np.ndarray,
    output_path: Path,
    channel_names: list[str] = None,
    apply_activation: bool = True,
):
    """
    Plot histogram of prediction intensities.
    
    Args:
        prediction: CZYX array
        output_path: Path to save histogram PNG
        channel_names: Names for channels (default: ['boundary'])
        apply_activation: Whether to apply sigmoid before plotting
    """
    if channel_names is None:
        channel_names = ['boundary']
    
    C, Z, Y, X = prediction.shape
    
    if apply_activation:
        prediction_activated = apply_sigmoid(prediction)
        title_suffix = " (after sigmoid)"
    else:
        prediction_activated = prediction
        title_suffix = " (logits)"
    
    fig, axes = plt.subplots(1, C, figsize=(6 * C, 5), squeeze=False)
    axes = axes[0]
    
    for c_idx, channel_name in enumerate(channel_names):
        ax = axes[c_idx]
        
        # Flatten channel data
        channel_data = prediction_activated[c_idx].flatten()
        
        # Plot histogram
        ax.hist(channel_data, bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{channel_name} intensity distribution{title_suffix}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = channel_data.mean()
        std_val = channel_data.std()
        median_val = np.median(channel_data)
        ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.4f} (±{std_val:.4f})')
        ax.axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.4f}')
        ax.legend()
        
        # Log scale for y-axis if needed
        if channel_data.max() / channel_data.min() > 1000:
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to: {output_path}")


def visualize_prediction_slices(
    prediction: np.ndarray,
    output_path: Path,
    num_slices: int = 5,
    channel_names: list[str] = None,
    apply_activation: bool = True,
):
    """
    Visualize multiple Z-slices from a prediction volume.
    
    Args:
        prediction: CZYX array (C=1: boundary)
        output_path: Path to save visualization PNG
        num_slices: Number of evenly-spaced slices to visualize
        channel_names: Names for channels (default: ['boundary'])
        apply_activation: Whether to apply sigmoid before visualization
    """
    if channel_names is None:
        channel_names = ['boundary']
    
    C, Z, Y, X = prediction.shape
    
    if C != len(channel_names):
        raise ValueError(f"Number of channels ({C}) doesn't match channel_names length ({len(channel_names)})")
    
    # Apply sigmoid if requested
    if apply_activation:
        prediction = apply_sigmoid(prediction)
    
    # Select evenly-spaced Z slices
    z_indices = np.linspace(0, Z - 1, num_slices, dtype=int)
    
    # Create figure with subplots: rows = channels, cols = slices
    fig, axes = plt.subplots(
        C, num_slices, 
        figsize=(4 * num_slices, 4 * C),
        squeeze=False
    )
    
    for c_idx, channel_name in enumerate(channel_names):
        for slice_idx, z_idx in enumerate(z_indices):
            ax = axes[c_idx, slice_idx]
            
            # Extract slice
            img = prediction[c_idx, z_idx]  # (Y, X)
            
            # Apply slight smoothing to reduce artifacts
            img = ndimage.gaussian_filter(img, sigma=0.5)
            
            # Normalize for visualization
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_norm = (img - img_min) / (img_max - img_min)
            else:
                img_norm = np.zeros_like(img)
            
            # Display
            ax.imshow(img_norm, cmap='viridis', interpolation='bilinear')  # Changed from 'nearest' to 'bilinear'
            ax.set_title(f'{channel_name}\nZ={z_idx}/{Z-1}')
            ax.axis('off')
            ax.grid(False)  # Explicitly disable grid
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {output_path}")


def visualize_middle_slice(
    prediction: np.ndarray,
    output_path: Path,
    channel_names: list[str] = None,
    apply_activation: bool = True,
):
    """
    Visualize the middle Z-slice for the boundary channel.
    
    Args:
        prediction: CZYX array (C=1: boundary)
        output_path: Path to save visualization PNG
        channel_names: Names for channels (default: ['boundary'])
        apply_activation: Whether to apply sigmoid before visualization
    """
    if channel_names is None:
        channel_names = ['boundary']
    
    # Apply sigmoid if requested
    if apply_activation:
        prediction = apply_sigmoid(prediction)
    
    C, Z, Y, X = prediction.shape
    z_mid = Z // 2
    
    fig, axes = plt.subplots(1, C, figsize=(6 * C, 6), squeeze=False)
    axes = axes[0]
    
    for c_idx, channel_name in enumerate(channel_names):
        ax = axes[c_idx]
        
        # Extract middle slice
        img = prediction[c_idx, z_mid]  # (Y, X)
        
        # Normalize for visualization
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img)
        
        # Display
        im = ax.imshow(img_norm, cmap='viridis', interpolation='bilinear')  # Changed from 'nearest'
        ax.set_title(f'{channel_name} prediction\nZ={z_mid}/{Z-1}')
        ax.axis('off')
        ax.grid(False)  # Explicitly disable grid
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Swin-UNETR inference outputs from Zarr or NPY files'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing prediction files (pred_*.zarr or pred_*.npy)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: input_dir/visualizations)'
    )
    parser.add_argument(
        '--num-slices',
        type=int,
        default=5,
        help='Number of Z-slices to visualize per volume (default: 5)'
    )
    parser.add_argument(
        '--middle-only',
        action='store_true',
        help='Only visualize middle slice (faster)'
    )
    parser.add_argument(
        '--max-volumes',
        type=int,
        default=None,
        help='Maximum number of volumes to visualize (default: all)'
    )
    parser.add_argument(
        '--no-activation',
        action='store_true',
        help='Do not apply sigmoid activation (show raw logits)'
    )
    parser.add_argument(
        '--no-histogram',
        action='store_true',
        help='Skip histogram plotting'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all prediction files (Zarr or NPY)
    prediction_files = sorted(
        list(input_dir.glob('pred_*.zarr')) + 
        list(input_dir.glob('pred_*.npy'))
    )
    
    if not prediction_files:
        print(f"No prediction files found in {input_dir}")
        print("Looking for files matching patterns: pred_*.zarr or pred_*.npy")
        return
    
    if args.max_volumes:
        prediction_files = prediction_files[:args.max_volumes]
    
    print(f"Found {len(prediction_files)} prediction files")
    
    apply_activation = not args.no_activation
    
    for prediction_file in prediction_files:
        print(f"\nProcessing: {prediction_file.name}")
        
        try:
            # Load prediction
            prediction = load_zarr_prediction(prediction_file)
            print(f"  Loaded shape: {prediction.shape}")
            
            # Generate histogram
            if not args.no_histogram:
                base_name = prediction_file.stem  # pred_<name>
                histogram_path = output_dir / f"{base_name}_histogram.png"
                plot_intensity_histogram(
                    prediction,
                    histogram_path,
                    apply_activation=apply_activation
                )
            
            # Generate visualization
            base_name = prediction_file.stem  # pred_<name>
            
            if args.middle_only:
                output_path = output_dir / f"{base_name}_middle_slice.png"
                visualize_middle_slice(
                    prediction, 
                    output_path,
                    apply_activation=apply_activation
                )
            else:
                output_path = output_dir / f"{base_name}_slices.png"
                visualize_prediction_slices(
                    prediction, 
                    output_path, 
                    num_slices=args.num_slices,
                    apply_activation=apply_activation
                )
        
        except Exception as e:
            print(f"  Error processing {prediction_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nVisualization complete! Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()

