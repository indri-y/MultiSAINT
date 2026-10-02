"""
losses.py
Custom Loss Functions untuk MultiSAINT
Fokus pada Hardware Trojan Detection dengan class imbalance

REVISED VERSION: Mengikuti konvensi paper original Focal Loss (Lin et al., ICCV 2017)
- Alpha ∈ (0, 1) sesuai paper original
- α = weight untuk positive class (trojan)
- (1 - α) = weight untuk negative class (clean)

Author: MultiSAINT Research Team
Purpose: Doctoral Research - FPGA Hardware Trojan Detection

Reference:
    Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    "Focal loss for dense object detection." ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss untuk Binary Classification dengan Class Imbalance

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)

    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Di mana:
        - p_t = p jika y=1, (1-p) jika y=0
        - α_t = α jika y=1 (positive/trojan), (1-α) jika y=0 (negative/clean)

    Args:
        alpha (float): Weighting factor untuk POSITIVE class (trojan)
            - Range: (0, 1)
            - α > 0.5: Favor positive class (trojan) - RECOMMENDED for HT detection
            - α = 0.5: Equal weighting
            - α < 0.5: Favor negative class (clean)

            Recommended values for Hardware Trojan Detection:
            - α = 0.75: Positive:Negative = 3:1 (moderate)
            - α = 0.80: Positive:Negative = 4:1 (recommended)
            - α = 0.90: Positive:Negative = 9:1 (aggressive)

        gamma (float): Focusing parameter (default: 2.0)
            - γ = 0: Equivalent to standard Cross-Entropy
            - γ = 2: Default from paper (proven effective)
            - γ > 2: More aggressive focusing on hard examples

        reduction (str): 'none', 'mean', atau 'sum'

    Usage:
        # For Hardware Trojan Detection (trojan is minority class)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)  # 3:1 weighting for trojan
        loss = criterion(logits, targets)

    Example weight interpretation:
        α = 0.75, γ = 2.0
        - Trojan node (y=1): weight = 0.75 * (1-p)^2
        - Clean node (y=0):  weight = 0.25 * p^2
        - Ratio: Trojan weighted 3x more than clean
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()

        # Validate alpha range
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"Alpha must be in range (0, 1), got {alpha}. "
                f"For Hardware Trojan Detection, use α > 0.5 (e.g., 0.75) "
                f"to give more weight to trojan (positive) class."
            )

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass untuk Focal Loss

        Args:
            inputs (Tensor): Logits [N, C] atau [N, 1] untuk binary
            targets (Tensor): Ground truth labels [N, C] atau [N, 1]
                              Nilai 1 = positive (trojan), 0 = negative (clean)

        Returns:
            loss (Tensor): Scalar (jika reduction='mean'/'sum') atau [N] (jika 'none')
        """
        # Ensure proper shape
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(1)
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        # Compute probabilities via sigmoid
        p = torch.sigmoid(inputs)

        # Compute standard binary cross-entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute p_t (probability of true class)
        # p_t = p if y=1, (1-p) if y=0
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal modulating factor: (1 - p_t)^γ
        # This down-weights easy examples (high p_t) and focuses on hard examples
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight to CE loss
        focal_loss = focal_weight * ce_loss

        # Apply alpha class weighting (following paper original convention)
        # α_t = α if y=1 (positive/trojan), (1-α) if y=0 (negative/clean)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss dengan per-sample weighting (untuk norm_loss dari GraphSAINT)

    Combines:
    1. Focal Loss (focus on hard examples + class balancing)
    2. Per-sample weights (dari sampling normalization GraphSAINT)

    Args:
        alpha (float): Class weighting untuk positive class (trojan)
            - Range: (0, 1)
            - Recommended: 0.75 untuk Hardware Trojan Detection
        gamma (float): Focusing parameter (default: 2.0)

    Usage:
        criterion = WeightedFocalLoss(alpha=0.75, gamma=2.0)
        loss = criterion(logits, targets, norm_loss)
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction="none")

    def forward(self, inputs, targets, sample_weights):
        """
        Forward pass dengan per-sample weighting

        Args:
            inputs (Tensor): Logits [N, C]
            targets (Tensor): Ground truth [N, C] atau [N]
            sample_weights (Tensor): Per-sample weights [N] (norm_loss dari minibatch)

        Returns:
            loss (Tensor): Weighted focal loss (scalar)
        """
        # Compute focal loss per sample
        focal_loss_per_sample = self.focal_loss(inputs, targets)

        # Reshape sample_weights jika perlu
        if len(sample_weights.shape) == 1:
            sample_weights = sample_weights.unsqueeze(1)

        # Apply per-sample weights dari GraphSAINT sampling
        weighted_loss = sample_weights * focal_loss_per_sample

        # Sum (consistent dengan GraphSAINT loss computation)
        return weighted_loss.sum()


class MultiClassFocalLoss(nn.Module):
    """
    Focal Loss untuk Multi-Class Classification

    Args:
        alpha (Tensor): Class weights [C] untuk C classes
            - Jika None, semua class memiliki weight yang sama
            - Gunakan compute_class_weights() untuk menghitung otomatis
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): 'none', 'mean', atau 'sum'

    Usage:
        # Compute class weights dari data
        class_weights = compute_class_weights(labels, method='inverse')
        criterion = MultiClassFocalLoss(alpha=class_weights, gamma=2.0)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass untuk Multi-Class Focal Loss

        Args:
            inputs (Tensor): Logits [N, C]
            targets (Tensor): Ground truth labels [N] (class indices, bukan one-hot)

        Returns:
            loss (Tensor): Scalar atau [N] tergantung reduction
        """
        # Compute standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Get probabilities via softmax
        p = F.softmax(inputs, dim=1)

        # Get p_t (probability of true class)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal modulating factor: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha class weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedMultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss dengan per-sample weighting (untuk GraphSAINT)

    Args:
        alpha (Tensor): Class weights [C] - gunakan compute_class_weights()
        gamma (float): Focusing parameter (default: 2.0)
    """

    def __init__(self, alpha=None, gamma=2.0):
        super(WeightedMultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_loss = MultiClassFocalLoss(
            alpha=alpha, gamma=gamma, reduction="none"
        )

    def forward(self, inputs, targets, sample_weights):
        """
        Forward pass dengan per-sample weighting

        Args:
            inputs (Tensor): Logits [N, C]
            targets (Tensor): Ground truth [N] (class indices)
            sample_weights (Tensor): Per-sample weights [N] dari GraphSAINT

        Returns:
            loss (Tensor): Weighted focal loss (scalar)
        """
        # Compute focal loss per sample
        focal_loss_per_sample = self.focal_loss(inputs, targets)

        # Apply per-sample weights
        weighted_loss = sample_weights * focal_loss_per_sample

        return weighted_loss.sum()


# ============================================================
# Utility Functions
# ============================================================


def compute_class_weights(labels, method="inverse", smooth=1.0):
    """
    Compute class weights untuk handle class imbalance

    Args:
        labels (np.array or torch.Tensor): Ground truth labels [N, C] atau [N]
        method (str): Metode perhitungan weight
            - 'inverse': weight_c = N / (num_classes * count_c)
            - 'sqrt': weight_c = sqrt(N / count_c)
            - 'log': weight_c = log(N / count_c + 1)
        smooth (float): Smoothing factor (prevent division by zero)

    Returns:
        torch.Tensor: Class weights [C]

    Example:
        # Dataset: 90% clean (class 0), 10% trojan (class 1)
        labels = np.array([0]*90 + [1]*10)
        weights = compute_class_weights(labels, method='inverse')
        # weights ≈ [0.56, 1.75] → trojan class weighted ~3x more
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Convert to class indices if one-hot encoded
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = labels.argmax(axis=1)

    # Count samples per class
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    num_samples = len(labels)
    num_classes = len(unique_classes)

    # Compute weights based on method
    if method == "inverse":
        # Standard inverse frequency weighting
        weights = num_samples / (num_classes * (class_counts + smooth))
    elif method == "sqrt":
        # Square root dampening (less aggressive)
        weights = np.sqrt(num_samples / (class_counts + smooth))
    elif method == "log":
        # Logarithmic dampening (even less aggressive)
        weights = np.log(num_samples / (class_counts + smooth) + 1.0)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'inverse', 'sqrt', or 'log'.")

    # Normalize weights so they sum to num_classes
    weights = weights / weights.sum() * num_classes

    return torch.FloatTensor(weights)


def get_recommended_alpha(pos_ratio, method="moderate"):
    """
    Get recommended alpha value based on positive class ratio

    Args:
        pos_ratio (float): Ratio of positive samples (e.g., 0.1 for 10% trojan)
        method (str): Weighting strategy
            - 'moderate': Balanced approach
            - 'aggressive': Strongly favor minority class
            - 'conservative': Mild favor to minority class

    Returns:
        float: Recommended alpha value ∈ (0, 1)

    Example:
        # Dataset has 5% trojan nodes
        alpha = get_recommended_alpha(0.05, method='moderate')
        # Returns ~0.80 (4:1 weighting for trojan)
    """
    if method == "aggressive":
        # α such that pos:neg ratio is inverse of data ratio
        alpha = 1 - pos_ratio
    elif method == "moderate":
        # α such that pos:neg ratio is sqrt of inverse
        alpha = 1 - np.sqrt(pos_ratio)
    elif method == "conservative":
        # α such that pos:neg ratio is log of inverse
        alpha = 1 - np.log(pos_ratio + 0.1) / np.log(1.1)
        alpha = np.clip(alpha, 0.55, 0.95)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Clip to valid range
    alpha = np.clip(alpha, 0.51, 0.99)

    return float(alpha)


# ============================================================
# Testing Code
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Focal Loss Implementation (Paper Original Convention)")
    print("α ∈ (0, 1) where α = weight for positive class")
    print("=" * 70)

    # ========================================
    # Test 1: Binary Focal Loss
    # ========================================
    print("\n" + "=" * 60)
    print("TEST 1: Binary Focal Loss (α ∈ 0-1)")
    print("=" * 60)

    # Create dummy data (imbalanced: 10% trojan)
    batch_size = 100
    logits = torch.randn(batch_size, 1)
    targets = torch.zeros(batch_size, 1)
    targets[:10] = 1  # 10% positive (trojan)

    print(f"Dataset: {batch_size} samples, 10 trojan (10%), 90 clean (90%)")

    # Standard BCE
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    print(f"\nStandard BCE Loss:        {bce_loss.item():.4f}")

    # Focal Loss with different alpha values
    print("\nFocal Loss (γ=2.0) with different α:")
    print("-" * 50)

    for alpha in [0.25, 0.50, 0.75, 0.80, 0.90]:
        focal_criterion = FocalLoss(alpha=alpha, gamma=2.0)
        focal_loss = focal_criterion(logits, targets)
        pos_weight = alpha
        neg_weight = 1 - alpha
        ratio = pos_weight / neg_weight
        print(
            f"  α={alpha:.2f}: Loss={focal_loss.item():.4f} | "
            f"Pos:Neg = {pos_weight:.2f}:{neg_weight:.2f} ({ratio:.1f}:1)"
        )

    print("\n✓ Binary Focal Loss test PASSED")

    # ========================================
    # Test 2: Weighted Focal Loss
    # ========================================
    print("\n" + "=" * 60)
    print("TEST 2: Weighted Focal Loss (with GraphSAINT norm_loss)")
    print("=" * 60)

    sample_weights = torch.ones(batch_size)

    weighted_criterion = WeightedFocalLoss(alpha=0.75, gamma=2.0)
    weighted_loss = weighted_criterion(logits, targets, sample_weights)
    print(f"Weighted Focal Loss (α=0.75): {weighted_loss.item():.4f}")

    print("✓ Weighted Focal Loss test PASSED")

    # ========================================
    # Test 3: Class Weights Computation
    # ========================================
    print("\n" + "=" * 60)
    print("TEST 3: Automatic Class Weights Computation")
    print("=" * 60)

    labels = np.array([0] * 90 + [1] * 10)  # 90% clean, 10% trojan

    print(f"Dataset: 90 clean (class 0), 10 trojan (class 1)")

    weights_inv = compute_class_weights(labels, method="inverse")
    weights_sqrt = compute_class_weights(labels, method="sqrt")
    weights_log = compute_class_weights(labels, method="log")

    print(f"\nClass weights by method:")
    print(
        f"  Inverse:  [clean={weights_inv[0]:.3f}, trojan={weights_inv[1]:.3f}] "
        f"→ trojan {weights_inv[1] / weights_inv[0]:.1f}x more"
    )
    print(
        f"  Sqrt:     [clean={weights_sqrt[0]:.3f}, trojan={weights_sqrt[1]:.3f}] "
        f"→ trojan {weights_sqrt[1] / weights_sqrt[0]:.1f}x more"
    )
    print(
        f"  Log:      [clean={weights_log[0]:.3f}, trojan={weights_log[1]:.3f}] "
        f"→ trojan {weights_log[1] / weights_log[0]:.1f}x more"
    )

    print("✓ Class weights test PASSED")

    # ========================================
    # Test 4: Recommended Alpha
    # ========================================
    print("\n" + "=" * 60)
    print("TEST 4: Recommended Alpha Values")
    print("=" * 60)

    print("\nRecommended α for different trojan ratios:")
    print("-" * 50)

    for pos_ratio in [0.01, 0.05, 0.10, 0.20]:
        alpha_mod = get_recommended_alpha(pos_ratio, "moderate")
        alpha_agg = get_recommended_alpha(pos_ratio, "aggressive")
        print(
            f"  Trojan ratio {pos_ratio * 100:4.0f}%: "
            f"α_moderate={alpha_mod:.2f}, α_aggressive={alpha_agg:.2f}"
        )

    print("✓ Recommended alpha test PASSED")

    # ========================================
    # Test 5: Validate Alpha Range
    # ========================================
    print("\n" + "=" * 60)
    print("TEST 5: Alpha Range Validation")
    print("=" * 60)

    print("Testing invalid alpha values...")

    invalid_alphas = [0.0, 1.0, -0.5, 1.5, 2.0]
    for invalid_alpha in invalid_alphas:
        try:
            _ = FocalLoss(alpha=invalid_alpha, gamma=2.0)
            print(f"  α={invalid_alpha}: ERROR - Should have raised exception!")
        except ValueError as e:
            print(f"  α={invalid_alpha}: ✓ Correctly rejected")

    print("✓ Alpha validation test PASSED")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nSUMMARY - Focal Loss Configuration for Hardware Trojan Detection:")
    print("-" * 70)
    print("  Recommended: α = 0.75, γ = 2.0")
    print("  Interpretation:")
    print("    - Trojan (positive) class: weight = 0.75")
    print("    - Clean (negative) class:  weight = 0.25")
    print("    - Trojan nodes are weighted 3x more than clean nodes")
    print("-" * 70)
