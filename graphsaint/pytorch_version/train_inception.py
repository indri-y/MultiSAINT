"""
train_inception.py - FULLY CORRECTED VERSION
MultiSAINT Training Script dengan Inception Architecture

CRITICAL CORRECTION APPLIED:
- [FIXED] Data Leakage: Best epoch now selected by VALIDATION F1 (not Test F1)

Other Features:
- Train accuracy dan F1 score dihitung dengan benar
- Train predictions dan labels diakumulasi selama epoch
- Train metrics dihitung setelah epoch selesai
- CSV flush for immediate write
- Proper config name detection
- All tensor to float/int conversion

Author: MultiSAINT Research Team
"""

from graphsaint.globals import *
from graphsaint.pytorch_version.models_inception import MultiSAINT_Inception
from graphsaint.pytorch_version.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from graphsaint.pytorch_version.utils import *

import torch
import time
import numpy as np
import csv
import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from graphsaint.pytorch_version.performance_metrics import (
    PerformanceMetricsTracker,
    print_performance_summary,
    save_performance_to_csv,
)


def calculate_metrics(labels, preds):
    """Calculate comprehensive metrics for binary classification"""
    from sklearn.metrics import confusion_matrix

    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(preds, axis=1)

    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    tpr = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, pos_label=1, zero_division=0)

    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "acc": float(acc),
        "prec": float(prec),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_full_batch(model, minibatch, mode="val"):
    """Full batch evaluation"""
    loss, preds, labels = model.eval_step(*minibatch.one_batch(mode=mode))

    if mode == "val":
        node_target = [minibatch.node_val]
    elif mode == "test":
        node_target = [minibatch.node_test]
    else:
        node_target = [minibatch.node_val, minibatch.node_test]

    f1mic, f1mac = [], []
    for n in node_target:
        f1_scores = calc_f1(to_numpy(labels[n]), to_numpy(preds[n]), model.sigmoid_loss)
        f1mic.append(f1_scores[0])
        f1mac.append(f1_scores[1])

    f1mic = f1mic[0] if len(f1mic) == 1 else f1mic
    f1mac = f1mac[0] if len(f1mac) == 1 else f1mac

    return loss, f1mic, f1mac, preds, labels


def prepare(train_data, train_params, arch_gcn):
    """Prepare data structures and initialize model"""
    adj_full, adj_train, feat_full, class_arr, role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)

    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)

    inception_type = arch_gcn.get("inception_type", "standard")

    model = MultiSAINT_Inception(
        num_classes,
        arch_gcn,
        train_params,
        feat_full,
        class_arr,
        inception_layer_cls=inception_type,
    )

    print("=" * 70)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("TOTAL PARAMETERS: {:,}".format(total_params))
    print("TRAINABLE PARAMETERS: {:,}".format(trainable_params))
    print("=" * 70)

    minibatch_eval = Minibatch(
        adj_full_norm, adj_train, role, train_params, cpu_eval=args_global.cpu_eval
    )
    model_eval = MultiSAINT_Inception(
        num_classes,
        arch_gcn,
        train_params,
        feat_full,
        class_arr,
        cpu_eval=args_global.cpu_eval,
        inception_layer_cls=inception_type,
    )

    if args_global.gpu >= 0:
        model = model.cuda()

    return model, minibatch, minibatch_eval, model_eval


def train(
    train_phases,
    model,
    minibatch,
    minibatch_eval,
    model_eval,
    eval_val_every,
    results_csv_path,
    seed,
    total_params,
    trainable_params,
):
    """Main training loop"""
    if not args_global.cpu_eval:
        minibatch_eval = minibatch

    epoch_ph_start = 0
    f1mic_best, ep_best = 0, -1
    time_train = 0
    cumulative_time = 0.0

    # Detect config ID from filename
    config_file = os.path.basename(args_global.train_config)

    if "order1" in config_file and "noattention" in config_file:
        config_id = "order1_noatt"
    elif "order1" in config_file and "attention" in config_file:
        config_id = "order1_att"
    elif "order2" in config_file and "noattention" in config_file:
        config_id = "order2_noatt"
    elif "order2" in config_file and "attention" in config_file:
        config_id = "order2_att"
    elif "order3" in config_file and "noattention" in config_file:
        config_id = "order3_noatt"
    elif "order3" in config_file and "attention" in config_file:
        config_id = "order3_att"
    else:
        config_id = "unknown"

    # Build paths
    model_name = "multisaint_{}_seed{}_best.pkl".format(config_id, seed)
    path_saver = os.path.join("results", "models", model_name)

    # Create directories
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(path_saver), exist_ok=True)

    print("\n" + "=" * 70)
    print("FILE PATHS FOR THIS RUN:")
    print("=" * 70)
    print("Config ID: {}".format(config_id))
    print("Seed: {}".format(seed))
    print("CSV: {}".format(results_csv_path))
    print("Model: {}".format(path_saver))
    print("=" * 70 + "\n")

    # Performance tracker
    perf_tracker = PerformanceMetricsTracker(
        model_eval, model_name="MultiSAINT_Inception"
    )

    # CSV columns
    csv_columns = [
        "epoch",
        "train_loss",
        "train_acc",
        "train_f1",
        "val_loss",
        "val_acc",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_tnr",
        "val_fpr",
        "val_fnr",
        "val_tn",
        "val_fp",
        "val_fn",
        "val_tp",
        "test_acc",
        "test_f1",
        "test_f1mic",
        "test_f1mac",
        "test_precision",
        "test_recall",
        "test_tnr",
        "test_fpr",
        "test_fnr",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",
        "gpu_utilization_pct",
        "peak_gpu_mem_mb",
        "gpu_mem_reserved_mb",
        "cpu_mem_mb",
        "time_per_epoch_sec",
        "cumulative_time_sec",
        "model_size_mb",
        "inference_latency_ms",
        "throughput_samples_per_sec",
        "cpu_runtime_mem_mb",
        "total_params",
        "trainable_params",
        "best_epoch",
    ]

    csv_file = open(results_csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()
    csv_file.flush()

    print("CSV logger initialized: {}".format(results_csv_path))

    for ip, phase in enumerate(train_phases):
        print("\n" + "=" * 70)
        print("TRAINING PHASE {}/{}".format(ip + 1, len(train_phases)))
        print("=" * 70)

        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()

        print("Sampler: {}".format(phase["sampler"]))
        print("Epochs: {} - {}".format(epoch_ph_start + 1, phase["end"]))
        print("Batches per epoch: ~{}".format(num_batches))

        for e in range(epoch_ph_start, phase["end"]):
            epoch_start_time = time.time()

            print(
                "\nEpoch {:3d}/{:3d} ".format(e + 1, phase["end"]), end="", flush=True
            )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            model.train()
            minibatch.shuffle()

            # ===== FIX: Akumulasi train predictions dan labels =====
            train_losses = []
            train_preds_all = []
            train_labels_all = []

            while not minibatch.end():
                loss_train, preds_train, labels_train = model.train_step(
                    *minibatch.one_batch(mode="train")
                )
                train_losses.append(loss_train.item())

                # Akumulasi predictions dan labels (detach untuk hemat memory)
                train_preds_all.append(preds_train.detach())
                train_labels_all.append(labels_train.detach())

            avg_train_loss = np.mean(train_losses) if len(train_losses) > 0 else 0.0

            # ===== FIX: Hitung training metrics =====
            if len(train_preds_all) > 0:
                # Concatenate all batches
                train_preds_concat = torch.cat(train_preds_all, dim=0)
                train_labels_concat = torch.cat(train_labels_all, dim=0)

                # Convert to numpy
                train_preds_np = to_numpy(train_preds_concat)
                train_labels_np = to_numpy(train_labels_concat)

                # Calculate metrics
                train_metrics = calculate_metrics(train_labels_np, train_preds_np)
            else:
                train_metrics = {
                    k: 0
                    for k in [
                        "acc",
                        "f1",
                        "prec",
                        "tpr",
                        "tnr",
                        "fpr",
                        "fnr",
                        "tn",
                        "fp",
                        "fn",
                        "tp",
                    ]
                }
            # =====================================================

            # Validation
            if (e + 1) % eval_val_every == 0 or e == phase["end"] - 1:
                model_eval.load_state_dict(model.state_dict())

                if not args_global.cpu_eval and args_global.gpu >= 0:
                    model_eval = model_eval.cuda()

                loss_val, f1mic_val, f1mac_val, val_preds, val_labels = (
                    evaluate_full_batch(model_eval, minibatch_eval, mode="val")
                )

                _, _, _, test_preds, test_labels = evaluate_full_batch(
                    model_eval, minibatch_eval, mode="test"
                )

                val_nodes = minibatch_eval.node_val
                val_preds_np = to_numpy(val_preds[val_nodes])
                val_labels_np = to_numpy(val_labels[val_nodes])
                val_metrics = calculate_metrics(val_labels_np, val_preds_np)

                test_nodes = minibatch_eval.node_test
                test_preds_np = to_numpy(test_preds[test_nodes])
                test_labels_np = to_numpy(test_labels[test_nodes])
                test_metrics = calculate_metrics(test_labels_np, test_preds_np)

                # Calculate Validation F1 for model selection (correct methodology)
                test_f1mic, test_f1mac = calc_f1(
                    test_labels_np, test_preds_np, model.sigmoid_loss
                )

                # Save model at best VALIDATION F1-score (not test - avoid data snooping)
                if f1mic_val > f1mic_best:
                    f1mic_best = f1mic_val
                    ep_best = e + 1
                    torch.save(model.state_dict(), path_saver)
                    print("  ✓ NEW BEST! Val F1={:.4f}".format(f1mic_best), end="")
            else:
                loss_val = 0
                val_metrics = {
                    k: 0
                    for k in [
                        "acc",
                        "f1",
                        "prec",
                        "tpr",
                        "tnr",
                        "fpr",
                        "fnr",
                        "tn",
                        "fp",
                        "fn",
                        "tp",
                    ]
                }
                test_metrics = {
                    k: 0
                    for k in [
                        "acc",
                        "f1",
                        "prec",
                        "tpr",
                        "tnr",
                        "fpr",
                        "fnr",
                        "tn",
                        "fp",
                        "fn",
                        "tp",
                    ]
                }

            # Resource usage
            if torch.cuda.is_available():
                gpu_mem_used = torch.cuda.memory_allocated() / (1024**2)
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (
                    1024**2
                )
                peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
                gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**2)
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = util.gpu
                except:
                    gpu_utilization = 0
            else:
                gpu_mem_used = 0
                gpu_mem_total = 0
                peak_gpu_mem = 0
                gpu_mem_reserved = 0
                gpu_utilization = 0

            import psutil

            process = psutil.Process(os.getpid())
            cpu_mem = process.memory_info().rss / (1024**2)

            time_per_epoch = time.time() - epoch_start_time
            cumulative_time += time_per_epoch

            # ===== FIX: Write train_acc dan train_f1 yang benar =====
            csv_writer.writerow(
                {
                    "epoch": int(e + 1),
                    "train_loss": float(avg_train_loss),
                    "train_acc": float(train_metrics["acc"]),  # ← FIXED!
                    "train_f1": float(train_metrics["f1"]),  # ← FIXED!
                    "val_loss": float(loss_val)
                    if isinstance(loss_val, (int, float))
                    else (float(loss_val.item()) if hasattr(loss_val, "item") else 0),
                    "val_acc": float(val_metrics["acc"]),
                    "val_f1": float(val_metrics["f1"]),
                    "val_precision": float(val_metrics["prec"]),
                    "val_recall": float(val_metrics["tpr"]),
                    "val_tnr": float(val_metrics["tnr"]),
                    "val_fpr": float(val_metrics["fpr"]),
                    "val_fnr": float(val_metrics["fnr"]),
                    "val_tn": int(val_metrics["tn"]),
                    "val_fp": int(val_metrics["fp"]),
                    "val_fn": int(val_metrics["fn"]),
                    "val_tp": int(val_metrics["tp"]),
                    "test_acc": float(test_metrics["acc"]),
                    "test_f1": float(test_metrics["f1"]),
                    "test_f1mic": 0,
                    "test_f1mac": 0,
                    "test_precision": float(test_metrics["prec"]),
                    "test_recall": float(test_metrics["tpr"]),
                    "test_tnr": float(test_metrics["tnr"]),
                    "test_fpr": float(test_metrics["fpr"]),
                    "test_fnr": float(test_metrics["fnr"]),
                    "test_tn": int(test_metrics["tn"]),
                    "test_fp": int(test_metrics["fp"]),
                    "test_fn": int(test_metrics["fn"]),
                    "test_tp": int(test_metrics["tp"]),
                    "gpu_mem_used_mb": float(gpu_mem_used),
                    "gpu_mem_total_mb": float(gpu_mem_total),
                    "gpu_utilization_pct": float(gpu_utilization),
                    "peak_gpu_mem_mb": float(peak_gpu_mem),
                    "gpu_mem_reserved_mb": float(gpu_mem_reserved),
                    "cpu_mem_mb": float(cpu_mem),
                    "time_per_epoch_sec": float(time_per_epoch),
                    "cumulative_time_sec": float(cumulative_time),
                    "model_size_mb": 0,
                    "inference_latency_ms": 0,
                    "throughput_samples_per_sec": 0,
                    "cpu_runtime_mem_mb": 0,
                    "total_params": 0,
                    "trainable_params": 0,
                    "best_epoch": 0,
                }
            )
            csv_file.flush()
            # =====================================================

            # Print progress
            if (e + 1) % eval_val_every == 0 or e == phase["end"] - 1:
                print(
                    " Train: Loss={:.4f} Acc={:.4f} F1={:.4f} | Val F1={:.4f} | Test F1={:.4f} | {:.1f}s".format(
                        avg_train_loss,
                        train_metrics["acc"],
                        train_metrics["f1"],
                        val_metrics["f1"],
                        test_metrics["f1"],
                        time_per_epoch,
                    )
                )
            else:
                print(
                    " Train: Loss={:.4f} Acc={:.4f} F1={:.4f} | {:.1f}s".format(
                        avg_train_loss,
                        train_metrics["acc"],
                        train_metrics["f1"],
                        time_per_epoch,
                    )
                )

            time_train = cumulative_time

        epoch_ph_start = phase["end"]

    # Final evaluation
    print("\n" + "=" * 70)
    print("LOADING BEST MODEL FOR FINAL EVALUATION")
    print("=" * 70)

    model_eval.load_state_dict(torch.load(path_saver))

    if not args_global.cpu_eval and args_global.gpu >= 0:
        model_eval = model_eval.cuda()

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    inference_start_time = time.time()

    loss_test, f1mic_both, f1mac_both, test_preds, test_labels = evaluate_full_batch(
        model_eval, minibatch_eval, mode="valtest"
    )

    f1mic_val, f1mic_test = f1mic_both
    f1mac_val, f1mac_test = f1mac_both

    test_nodes = minibatch_eval.node_test
    test_preds_np = to_numpy(test_preds[test_nodes])
    test_labels_np = to_numpy(test_labels[test_nodes])
    test_metrics = calculate_metrics(test_labels_np, test_preds_np)

    inference_end_time = time.time()
    total_inference_time = inference_end_time - inference_start_time

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(
        "Validation (Epoch {}): F1_Micro = {:.4f} | F1_Macro = {:.4f}".format(
            ep_best, f1mic_val, f1mac_val
        )
    )
    print("Test Set Statistics:")
    print("  F1_Micro = {:.4f} | F1_Macro = {:.4f}".format(f1mic_test, f1mac_test))
    print(
        "  Accuracy = {:.4f} | Precision = {:.4f}".format(
            test_metrics["acc"], test_metrics["prec"]
        )
    )
    print(
        "  TPR      = {:.4f} | TNR       = {:.4f}".format(
            test_metrics["tpr"], test_metrics["tnr"]
        )
    )
    print(
        "  FPR      = {:.4f} | FNR       = {:.4f}".format(
            test_metrics["fpr"], test_metrics["fnr"]
        )
    )
    print("  F1-Score = {:.4f}".format(test_metrics["f1"]))
    print("Confusion Matrix:")
    print("  TN = {:5d} | FP = {:5d}".format(test_metrics["tn"], test_metrics["fp"]))
    print("  FN = {:5d} | TP = {:5d}".format(test_metrics["fn"], test_metrics["tp"]))
    print(
        "Total Training Time: {:.2f} sec ({:.2f} min)".format(
            time_train, time_train / 60
        )
    )
    print("=" * 70 + "\n")

    # Measure performance metrics
    print("\n" + "=" * 70)
    print("MEASURING MODEL EFFICIENCY METRICS")
    print("=" * 70)

    print("  Measuring model size...")
    model_size_info = perf_tracker.get_model_size_mb()
    model_size_mb = model_size_info["size_mb"]
    print("    Model size: {:.2f} MB".format(model_size_mb))

    print("  Preparing test batch...")
    test_batch = minibatch_eval.one_batch(mode="test")
    node_subgraph, adj_subgraph, norm_loss = test_batch
    print("    Test batch size: {} nodes".format(len(node_subgraph)))

    # Move to correct device
    if args_global.cpu_eval:
        if adj_subgraph.is_cuda:
            adj_subgraph = adj_subgraph.cpu()
        if norm_loss.is_cuda:
            norm_loss = norm_loss.cpu()
        print("    Device: CPU (cpu_eval mode)")
    else:
        if model_eval.use_cuda:
            if not adj_subgraph.is_cuda:
                adj_subgraph = adj_subgraph.cuda()
            if not norm_loss.is_cuda:
                norm_loss = norm_loss.cuda()
            print("    Device: GPU")
        else:
            if adj_subgraph.is_cuda:
                adj_subgraph = adj_subgraph.cpu()
            if norm_loss.is_cuda:
                norm_loss = norm_loss.cpu()
            print("    Device: CPU")

    print("  Measuring inference performance...")
    try:
        inference_metrics = perf_tracker.measure_inference_throughput(
            node_subgraph, adj_subgraph, norm_loss, num_iterations=100
        )
        latency_ms = inference_metrics.get("latency_ms_per_sample", 0)
        throughput = inference_metrics.get("throughput_samples_per_sec", 0)
        print("    Latency: {:.4f} ms/sample".format(latency_ms))
        print("    Throughput: {:.2f} samples/sec".format(throughput))
    except Exception as e:
        print("    Warning: Could not measure inference: {}".format(str(e)))
        latency_ms = 0
        throughput = 0

    print("  Measuring memory footprint...")
    try:
        memory_metrics = perf_tracker.get_memory_footprint()
        if "cpu_rss_mb" in memory_metrics:
            cpu_runtime_mem = memory_metrics["cpu_rss_mb"]
        elif "gpu_allocated_mb" in memory_metrics:
            cpu_runtime_mem = memory_metrics["gpu_allocated_mb"]
        else:
            cpu_runtime_mem = 0
        print("    Runtime memory: {:.2f} MB".format(cpu_runtime_mem))
    except Exception as e:
        print("    Warning: Could not measure memory: {}".format(str(e)))
        cpu_runtime_mem = 0

    print("=" * 70)

    # Write FINAL_TEST row
    csv_writer.writerow(
        {
            "epoch": "FINAL_TEST",
            "train_loss": 0,
            "train_acc": 0,
            "train_f1": 0,
            "val_loss": 0,
            "val_acc": 0,
            "val_f1": 0,
            "val_precision": 0,
            "val_recall": 0,
            "val_tnr": 0,
            "val_fpr": 0,
            "val_fnr": 0,
            "val_tn": 0,
            "val_fp": 0,
            "val_fn": 0,
            "val_tp": 0,
            "test_acc": float(test_metrics["acc"]),
            "test_f1": float(test_metrics["f1"]),
            "test_f1mic": float(f1mic_test),
            "test_f1mac": float(f1mac_test),
            "test_precision": float(test_metrics["prec"]),
            "test_recall": float(test_metrics["tpr"]),
            "test_tnr": float(test_metrics["tnr"]),
            "test_fpr": float(test_metrics["fpr"]),
            "test_fnr": float(test_metrics["fnr"]),
            "test_tn": int(test_metrics["tn"]),
            "test_fp": int(test_metrics["fp"]),
            "test_fn": int(test_metrics["fn"]),
            "test_tp": int(test_metrics["tp"]),
            "gpu_mem_used_mb": 0,
            "gpu_mem_total_mb": 0,
            "gpu_utilization_pct": 0,
            "peak_gpu_mem_mb": 0,
            "gpu_mem_reserved_mb": 0,
            "cpu_mem_mb": 0,
            "time_per_epoch_sec": 0,
            "cumulative_time_sec": float(time_train),
            "model_size_mb": float(model_size_mb),
            "inference_latency_ms": float(latency_ms),
            "throughput_samples_per_sec": float(throughput),
            "cpu_runtime_mem_mb": float(cpu_runtime_mem),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "best_epoch": int(ep_best),
        }
    )
    csv_file.flush()

    csv_file.close()

    print("\nTraining results saved to: {}".format(results_csv_path))
    print("Performance metrics included in CSV")

    # Performance summary
    print("\n" + "=" * 70)
    print("COMPILING PERFORMANCE SUMMARY")
    print("=" * 70)

    perf_summary = perf_tracker.get_summary(inference_metrics)

    perf_summary.update(
        {
            "train_time_sec": time_train,
            "inference_time_sec": total_inference_time,
            "total_time_sec": time_train + total_inference_time,
            "seed": seed,
            "inception_type": model_eval.inception_type,
            "max_order": model_eval.max_order_inception,
            "num_layers": model_eval.num_layers,
            "test_accuracy": test_metrics["acc"],
            "test_precision": test_metrics["prec"],
            "test_tpr": test_metrics["tpr"],
            "test_tnr": test_metrics["tnr"],
            "test_fpr": test_metrics["fpr"],
            "test_fnr": test_metrics["fnr"],
            "test_f1": test_metrics["f1"],
            "test_f1mic": f1mic_test,
            "test_f1mac": f1mac_test,
            "test_tn": test_metrics["tn"],
            "test_fp": test_metrics["fp"],
            "test_fn": test_metrics["fn"],
            "test_tp": test_metrics["tp"],
        }
    )

    print_performance_summary(perf_summary)

    print("\n" + "=" * 70)
    print("ALL METRICS SAVED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    log_dir(
        args_global.train_config,
        args_global.data_prefix,
        git_branch,
        git_rev,
        timestamp,
    )

    print("\nRandom seed set to: {}".format(args_global.seed))

    torch.manual_seed(args_global.seed)
    np.random.seed(args_global.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args_global.seed)

    print("\n" + "=" * 70)
    print("LOADING CONFIGURATION AND DATA")
    print("=" * 70)

    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)

    if "eval_val_every" not in train_params:
        train_params["eval_val_every"] = EVAL_VAL_EVERY_EP

    print("Configuration loaded successfully")

    print("\n" + "=" * 70)
    print("INITIALIZING MODEL AND MINIBATCH HANDLERS")
    print("=" * 70)

    model, minibatch, minibatch_eval, model_eval = prepare(
        train_data, train_params, arch_gcn
    )

    # Get parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model initialized successfully")

    # Build CSV path
    config_file = os.path.basename(args_global.train_config)

    if "order1" in config_file and "noattention" in config_file:
        config_id = "order1_noatt"
    elif "order1" in config_file and "attention" in config_file:
        config_id = "order1_att"
    elif "order2" in config_file and "noattention" in config_file:
        config_id = "order2_noatt"
    elif "order2" in config_file and "attention" in config_file:
        config_id = "order2_att"
    elif "order3" in config_file and "noattention" in config_file:
        config_id = "order3_noatt"
    elif "order3" in config_file and "attention" in config_file:
        config_id = "order3_att"
    else:
        config_id = "unknown"

    csv_name = "multisaint_{}_seed{}.csv".format(config_id, args_global.seed)
    results_csv_path = os.path.join("results", "csv", csv_name)

    print("Config ID: {}".format(config_id))
    print("Results will be saved to: {}".format(results_csv_path))

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    train(
        train_phases,
        model,
        minibatch,
        minibatch_eval,
        model_eval,
        train_params["eval_val_every"],
        results_csv_path,
        args_global.seed,
        total_params,
        trainable_params,
    )

    print("\n" + "=" * 70)
    print("TRAINING SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")
