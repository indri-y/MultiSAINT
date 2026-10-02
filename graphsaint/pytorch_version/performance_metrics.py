"""
performance_metrics.py
Performance Metrics Tracker for MultiSAINT (FP32 Baseline)
Tracks: Model Size, Memory, Throughput, Latency
For future comparison with Quantized models

Author: MultiSAINT Research Team
Purpose: Prepare baseline metrics for quantization comparison
"""

import torch
import time
import numpy as np
import os
import psutil


class PerformanceMetricsTracker:
    """
    Track comprehensive performance metrics for FP32 model
    Prepares baseline data for quantization comparison

    Tracks:
        - Model size (MB)
        - Parameter counts
        - Inference throughput (samples/sec)
        - Inference latency (ms/sample)
        - Memory footprint (GPU/CPU)
    """

    def __init__(self, model, model_name="MultiSAINT_FP32"):
        """
        Initialize performance tracker

        Args:
            model: PyTorch model (MultiSAINT)
            model_name: Name for logging/identification
        """
        self.model = model
        self.model_name = model_name
        self.device = next(model.parameters()).device

    def get_model_size_mb(self):
        """
        Calculate model size in MB (FP32)

        Returns:
            dict: {
                'size_mb': float,              # Total size in MB
                'total_params': int,           # Total parameters
                'trainable_params': int,       # Trainable parameters
                'param_size_bytes': int,       # Parameter memory (bytes)
                'buffer_size_bytes': int,      # Buffer memory (bytes)
                'layer_breakdown': dict        # Breakdown by layer type
            }
        """
        param_size = 0
        buffer_size = 0
        trainable_params = 0
        total_params = 0

        # Calculate parameter sizes
        for param in self.model.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            param_size += num_params * param.element_size()

        # Calculate buffer sizes (BatchNorm running stats, etc.)
        for buffer in self.model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        total_size_bytes = param_size + buffer_size
        size_mb = total_size_bytes / (1024**2)

        # Breakdown by layer type
        breakdown = self._get_layer_breakdown()

        return {
            "size_mb": float(size_mb),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "param_size_bytes": int(param_size),
            "buffer_size_bytes": int(buffer_size),
            "layer_breakdown": breakdown,
        }

    def _get_layer_breakdown(self):
        """
        Get parameter breakdown by layer type

        Returns:
            dict: {layer_type: {'count': int, 'params': int}}
        """
        breakdown = {}

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                num_params = sum(p.numel() for p in module.parameters())
                layer_type = "Linear"
            elif isinstance(module, torch.nn.BatchNorm1d):
                num_params = sum(p.numel() for p in module.parameters())
                layer_type = "BatchNorm1d"
            elif isinstance(module, torch.nn.Dropout):
                continue  # Dropout has no parameters
            else:
                continue

            if layer_type not in breakdown:
                breakdown[layer_type] = {"count": 0, "params": 0}

            breakdown[layer_type]["count"] += 1
            breakdown[layer_type]["params"] += num_params

        return breakdown

    def measure_inference_throughput(
        self, node_subgraph, adj_subgraph, norm_loss, num_iterations=100
    ):
        """
        Measure inference throughput and latency

        Args:
            node_subgraph: Node indices for inference (np.array or torch.Tensor)
            adj_subgraph: Adjacency matrix (torch.sparse)
            norm_loss: Normalization loss weights (torch.Tensor)
            num_iterations: Number of iterations for averaging (default: 100)

        Returns:
            dict: {
                'throughput_samples_per_sec': float,  # Samples processed per second
                'latency_ms_per_sample': float,       # Time per sample (ms)
                'avg_batch_time_ms': float,           # Average batch time (ms)
                'std_batch_time_ms': float,           # Std dev batch time (ms)
                'batch_size': int,                    # Batch size used
                'num_iterations': int                 # Number of iterations
            }
        """
        self.model.eval()

        times = []
        batch_size = len(node_subgraph)

        # Warmup runs (important for GPU to reach steady state)
        print("      Warmup (5 iterations)...", end=" ")
        with torch.no_grad():
            for _ in range(5):
                _ = self.model.eval_step(node_subgraph, adj_subgraph, norm_loss)
        print("Done")

        # Actual measurement
        print(f"      Measuring ({num_iterations} iterations)...", end=" ")
        with torch.no_grad():
            for i in range(num_iterations):
                # Synchronize CUDA before timing (critical for accurate measurement)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.perf_counter()

                # Run inference
                _ = self.model.eval_step(node_subgraph, adj_subgraph, norm_loss)

                # Synchronize CUDA after timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                times.append(end_time - start_time)

        print("Done")

        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)

        # Calculate metrics
        avg_batch_time_ms = avg_time * 1000
        std_batch_time_ms = std_time * 1000
        latency_ms_per_sample = avg_batch_time_ms / batch_size
        throughput_samples_per_sec = batch_size / avg_time

        return {
            "throughput_samples_per_sec": float(throughput_samples_per_sec),
            "latency_ms_per_sample": float(latency_ms_per_sample),
            "avg_batch_time_ms": float(avg_batch_time_ms),
            "std_batch_time_ms": float(std_batch_time_ms),
            "batch_size": int(batch_size),
            "num_iterations": int(num_iterations),
        }

    def get_memory_footprint(self):
        """
        Get current memory footprint

        Returns:
            dict: Memory usage in MB (GPU or CPU depending on device)
        """
        if torch.cuda.is_available() and self.device.type == "cuda":
            # GPU memory
            allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**2)

            return {
                "gpu_allocated_mb": float(allocated),
                "gpu_reserved_mb": float(reserved),
                "gpu_max_allocated_mb": float(max_allocated),
            }
        else:
            # CPU memory
            process = psutil.Process()
            mem_info = process.memory_info()

            return {
                "cpu_rss_mb": float(mem_info.rss / (1024**2)),
                "cpu_vms_mb": float(mem_info.vms / (1024**2)),
            }

    def get_summary(self, inference_metrics=None):
        """
        Get comprehensive performance summary

        Args:
            inference_metrics: Optional dict from measure_inference_throughput()

        Returns:
            dict: Complete performance summary with all tracked metrics
        """
        summary = {
            "model_name": self.model_name,
            "precision": "FP32",
            "device": str(self.device),
        }

        # Model size
        model_size = self.get_model_size_mb()
        summary.update(
            {
                "model_size_mb": model_size["size_mb"],
                "total_params": model_size["total_params"],
                "trainable_params": model_size["trainable_params"],
                "param_size_bytes": model_size["param_size_bytes"],
                "buffer_size_bytes": model_size["buffer_size_bytes"],
            }
        )

        # Layer breakdown (optional, can be verbose)
        # summary['layer_breakdown'] = model_size['layer_breakdown']

        # Memory footprint
        memory = self.get_memory_footprint()
        summary.update(memory)

        # Inference metrics (if provided)
        if inference_metrics:
            summary.update(inference_metrics)

        return summary


def print_performance_summary(summary):
    """
    Print performance summary in readable format with nice formatting

    Args:
        summary: dict from PerformanceMetricsTracker.get_summary()
    """
    print("\n" + "=" * 80)
    print(f"PERFORMANCE SUMMARY: {summary['model_name']}")
    print("=" * 80)

    print(f"\n📊 MODEL CONFIGURATION:")
    print(f"  Precision:        {summary['precision']}")
    print(f"  Device:           {summary['device']}")
    print(f"  Model Size:       {summary['model_size_mb']:.2f} MB")
    print(f"  Total Parameters: {summary['total_params']:,}")
    print(f"  Trainable Params: {summary['trainable_params']:,}")

    # Additional detail
    if "param_size_bytes" in summary:
        print(f"  Param Memory:     {summary['param_size_bytes'] / (1024**2):.2f} MB")
        print(f"  Buffer Memory:    {summary['buffer_size_bytes'] / (1024**2):.2f} MB")

    print(f"\n💾 MEMORY UTILIZATION:")
    if "gpu_allocated_mb" in summary:
        print(f"  GPU Allocated:    {summary['gpu_allocated_mb']:.2f} MB")
        print(f"  GPU Reserved:     {summary['gpu_reserved_mb']:.2f} MB")
        print(f"  GPU Max Alloc:    {summary['gpu_max_allocated_mb']:.2f} MB")
    else:
        print(f"  CPU RSS:          {summary.get('cpu_rss_mb', 0):.2f} MB")
        print(f"  CPU VMS:          {summary.get('cpu_vms_mb', 0):.2f} MB")

    if "throughput_samples_per_sec" in summary:
        print(f"\n⚡ COMPUTATIONAL SPEED:")
        print(
            f"  Throughput:       {summary['throughput_samples_per_sec']:.2f} samples/sec"
        )
        print(f"  Latency:          {summary['latency_ms_per_sample']:.2f} ms/sample")
        print(
            f"  Avg Batch Time:   {summary['avg_batch_time_ms']:.2f} ± {summary.get('std_batch_time_ms', 0):.2f} ms"
        )
        print(f"  Batch Size:       {summary.get('batch_size', 'N/A')}")

    # Accuracy metrics (if available)
    if "test_accuracy" in summary:
        print(f"\n🎯 ACCURACY METRICS:")
        print(f"  Test Accuracy:    {summary['test_accuracy']:.4f}")
        print(f"  Test Precision:   {summary['test_precision']:.4f}")
        print(f"  Test TPR:         {summary['test_tpr']:.4f}")
        print(f"  Test TNR:         {summary['test_tnr']:.4f}")
        print(f"  Test F1:          {summary['test_f1']:.4f}")

    # Timing
    if "train_time_sec" in summary:
        print(f"\n⏱️  TIMING:")
        print(
            f"  Training Time:    {summary['train_time_sec']:.2f} sec ({summary['train_time_sec'] / 60:.2f} min)"
        )
        print(f"  Inference Time:   {summary['inference_time_sec']:.2f} sec")
        print(
            f"  Total Time:       {summary['total_time_sec']:.2f} sec ({summary['total_time_sec'] / 60:.2f} min)"
        )

    print("=" * 80 + "\n")


def save_performance_to_csv(summary, csv_path):
    """
    Save performance summary to CSV (append mode for aggregation)

    Args:
        summary: dict from PerformanceMetricsTracker.get_summary()
        csv_path: Output CSV file path
    """
    import csv

    # Create directory if needed
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # Check if file exists
    file_exists = os.path.isfile(csv_path)

    # Open in append mode
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(summary)

    print(f"   ✅ Performance metrics saved to: {csv_path}")


def compare_fp32_vs_int8(fp32_summary, int8_summary):
    """
    Compare FP32 baseline vs INT8 quantized model
    (For future use when quantization is implemented)

    Args:
        fp32_summary: dict from FP32 model
        int8_summary: dict from INT8 model

    Returns:
        dict: Comparison metrics
    """
    comparison = {
        "model_size_reduction": 0.0,
        "throughput_improvement": 0.0,
        "latency_improvement": 0.0,
        "accuracy_drop": 0.0,
        "tpr_drop": 0.0,
        "tnr_drop": 0.0,
        "f1_drop": 0.0,
    }

    # Size comparison
    if "model_size_mb" in fp32_summary and "model_size_mb" in int8_summary:
        fp32_size = fp32_summary["model_size_mb"]
        int8_size = int8_summary["model_size_mb"]
        comparison["model_size_reduction"] = (fp32_size - int8_size) / fp32_size * 100

    # Speed comparison
    if (
        "throughput_samples_per_sec" in fp32_summary
        and "throughput_samples_per_sec" in int8_summary
    ):
        fp32_thr = fp32_summary["throughput_samples_per_sec"]
        int8_thr = int8_summary["throughput_samples_per_sec"]
        comparison["throughput_improvement"] = (int8_thr - fp32_thr) / fp32_thr * 100

    if (
        "latency_ms_per_sample" in fp32_summary
        and "latency_ms_per_sample" in int8_summary
    ):
        fp32_lat = fp32_summary["latency_ms_per_sample"]
        int8_lat = int8_summary["latency_ms_per_sample"]
        comparison["latency_improvement"] = (fp32_lat - int8_lat) / fp32_lat * 100

    # Accuracy comparison
    for metric in ["accuracy", "tpr", "tnr", "f1"]:
        key = f"test_{metric}"
        if key in fp32_summary and key in int8_summary:
            fp32_val = fp32_summary[key]
            int8_val = int8_summary[key]
            drop = fp32_val - int8_val
            comparison[f"{metric}_drop"] = drop * 100  # as percentage

    return comparison


def print_comparison(comparison):
    """
    Print FP32 vs INT8 comparison (for future use)

    Args:
        comparison: dict from compare_fp32_vs_int8()
    """
    print("\n" + "=" * 80)
    print("FP32 vs INT8 QUANTIZATION COMPARISON")
    print("=" * 80)

    print(f"\n📉 MODEL SIZE:")
    print(f"  Reduction: {comparison['model_size_reduction']:.2f}%")

    print(f"\n⚡ SPEED IMPROVEMENT:")
    print(f"  Throughput: +{comparison['throughput_improvement']:.2f}%")
    print(f"  Latency:    -{comparison['latency_improvement']:.2f}%")

    print(f"\n📊 ACCURACY IMPACT:")
    print(f"  Accuracy Drop: {comparison['accuracy_drop']:.2f}%")
    print(f"  TPR Drop:      {comparison['tpr_drop']:.2f}%")
    print(f"  TNR Drop:      {comparison['tnr_drop']:.2f}%")
    print(f"  F1 Drop:       {comparison['f1_drop']:.2f}%")

    print("=" * 80 + "\n")


# ============================================================
# ===== HELPER FUNCTIONS =====
# ============================================================


def get_model_flops(model, input_shape):
    """
    Estimate FLOPs for model (approximate)
    Note: This is a simplified estimation

    Args:
        model: PyTorch model
        input_shape: Tuple of input dimensions

    Returns:
        int: Estimated FLOPs
    """
    # This is a placeholder - full FLOP counting requires tools like thop or fvcore
    # For now, we estimate based on parameter count
    total_params = sum(p.numel() for p in model.parameters())

    # Rough estimate: 2 FLOPs per parameter per forward pass
    estimated_flops = total_params * 2

    return estimated_flops


def format_bytes(bytes_val):
    """
    Format bytes into human-readable string

    Args:
        bytes_val: Number of bytes

    Returns:
        str: Formatted string (e.g., "45.23 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"
