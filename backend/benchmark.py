#!/usr/bin/env python
import os
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Set matplotlib backend to Agg for headless environments
plt.switch_backend('agg')

def run_benchmark(model_type, test_images, iterations=5, warm_up=2, upscale=False):
    """Run benchmark on specified model type with test images.
    
    Args:
        model_type: 'pytorch' or 'onnx'
        test_images: List of image paths to test
        iterations: Number of iterations to run for each image
        warm_up: Number of warm-up iterations (not counted in timing)
        upscale: Whether to upscale the output image
    
    Returns:
        Dictionary with timing results
    """
    results = {
        'images': [],
        'sizes': [],
        'times': [],
        'model_type': model_type
    }
    
    # Import modules here to avoid issues with variable usage before assignment
    import torch
    import sys
    
    # Load appropriate model
    if model_type.lower() == 'pytorch':
        from inference import RainRemovalInference
        model = RainRemovalInference(
            model_name="proposed",
            ckpt_dir="checkpoint",
            ckpt_epoch=800,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            resize='original'
        )
    elif model_type.lower() == 'onnx':
        try:
            # Import ONNX model
            sys.path.append('onnx_models')
            from onnx_inference import ONNXRainRemovalInference
            model = ONNXRainRemovalInference(
                model_name="proposed",
                ckpt_dir="onnx_models",
                device='CUDA' if torch.cuda.is_available() else 'CPU',
                resize='original'
            )
        except ImportError:
            print("Error: ONNX model not found. Please convert the model first.")
            return None
    else:
        print(f"Unknown model type: {model_type}")
        return None
    
    # Test each image
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image: {img_path}")
            continue
        
        h, w = img.shape[:2]
        img_size = f"{w}x{h}"
        img_name = os.path.basename(img_path)
        
        print(f"Processing {img_name} ({img_size}) with {model_type} model...")
        
        # Warm-up runs
        for _ in range(warm_up):
            _ = model.process_image(img, output_size='upscale' if upscale else None)
        
        # Timed runs
        times = []
        for i in range(iterations):
            start_time = time.time()
            result = model.process_image(img, output_size='upscale' if upscale else None)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Iteration {i+1}/{iterations}: {elapsed:.3f}s")
        
        avg_time = sum(times) / len(times)
        results['images'].append(img_name)
        results['sizes'].append(img_size)
        results['times'].append(avg_time)
        
        print(f"  Average time: {avg_time:.3f}s")
    
    return results

def plot_comparison(pytorch_results, onnx_results, output_path):
    """Generate a comparison bar chart of PyTorch vs ONNX performance."""
    if pytorch_results is None or onnx_results is None:
        print("Cannot generate comparison plot: missing results")
        return
    
    # Check if images match
    common_images = []
    for i, img in enumerate(pytorch_results['images']):
        if img in onnx_results['images']:
            onnx_idx = onnx_results['images'].index(img)
            common_images.append((i, onnx_idx, img))
    
    if not common_images:
        print("No common images to compare")
        return
    
    # Create bar chart data
    image_names = [img for _, _, img in common_images]
    pytorch_times = [pytorch_results['times'][i] for i, _, _ in common_images]
    onnx_times = [onnx_results['times'][j] for _, j, _ in common_images]
    
    # Calculate speedup factors
    speedups = [p/o for p, o in zip(pytorch_times, onnx_times)]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Bar chart of processing times
    x = range(len(image_names))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], pytorch_times, width, label='PyTorch')
    ax1.bar([i + width/2 for i in x], onnx_times, width, label='ONNX')
    
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('PyTorch vs ONNX Processing Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(image_names, rotation=45, ha='right')
    ax1.legend()
    
    # Bar chart of speedup
    ax2.bar(x, speedups, 0.6)
    ax2.set_ylabel('Speedup Factor (PyTorch/ONNX)')
    ax2.set_title('ONNX Speedup Factor')
    ax2.set_xticks(x)
    ax2.set_xticklabels(image_names, rotation=45, ha='right')
    ax2.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
    
    # Add text labels for speedup values
    for i, v in enumerate(speedups):
        ax2.text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    import sys
    import torch
    
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX model performance")
    parser.add_argument("--image_dir", type=str, default="test_images", help="Directory with test images")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for each image")
    parser.add_argument("--warm_up", type=int, default=2, help="Number of warm-up iterations")
    parser.add_argument("--upscale", action="store_true", help="Test with upscaling enabled")
    parser.add_argument("--output", type=str, default="benchmark_results.png", help="Output plot filename")
    
    args = parser.parse_args()
    
    # Find test images
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    test_images = []
    for ext in [".jpg", ".jpeg", ".png"]:
        test_images.extend(list(Path(args.image_dir).glob(f"*{ext}")))
    
    test_images = [str(p) for p in test_images]
    if not test_images:
        print(f"Error: No images found in {args.image_dir}")
        sys.exit(1)
    
    print(f"Found {len(test_images)} test images")
    
    # Run benchmarks
    pytorch_results = run_benchmark('pytorch', test_images, args.iterations, args.warm_up, args.upscale)
    onnx_results = run_benchmark('onnx', test_images, args.iterations, args.warm_up, args.upscale)
    
    # Generate comparison plot
    plot_comparison(pytorch_results, onnx_results, args.output)
    
    # Print summary
    if pytorch_results and onnx_results:
        pytorch_avg = sum(pytorch_results['times']) / len(pytorch_results['times'])
        onnx_avg = sum(onnx_results['times']) / len(onnx_results['times'])
        speedup = pytorch_avg / onnx_avg
        
        print("\nBenchmark Summary:")
        print(f"  PyTorch average: {pytorch_avg:.3f}s")
        print(f"  ONNX average: {onnx_avg:.3f}s")
        print(f"  Overall speedup: {speedup:.2f}x")
    
    print("\nBenchmark complete!") 