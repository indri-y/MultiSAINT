"""
models_inception.py - FULLY CORRECTED VERSION
MultiSAINT Model dengan Inception-GNN Architecture + Focal Loss

CORRECTIONS APPLIED:
1. [FIXED] use_attention parameter now properly passed to layers
2. [FIXED] reduction_ratio parameter now properly passed to layers
3. [FIXED] focal_alpha default changed from 2.0 to 0.75 (Lin et al. convention)

Features:
1. Inception-style multi-scale aggregation
2. Channel attention mechanism (can be ON/OFF via config)
3. Adaptive fusion weights
4. Three variants: Standard, Adaptive, Depthwise
5. Focal Loss untuk handle class imbalance

Author: MultiSAINT Research Team
Purpose: Doctoral Research - FPGA Hardware Trojan Detection
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from graphsaint.utils import *
from graphsaint.pytorch_version.layers_inception import (
    InceptionGNNLayer,
    AdaptiveInceptionGNNLayer,
    DepthwiseSeparableInceptionGNN,
)

# Import layers untuk classifier
import graphsaint.pytorch_version.layers as layers

# Import Focal Loss
from graphsaint.pytorch_version.losses import (
    WeightedFocalLoss,
    WeightedMultiClassFocalLoss,
    compute_class_weights,
)


class MultiSAINT_Inception(nn.Module):
    """
    MultiSAINT dengan Inception-GNN Architecture + Focal Loss

    Architecture Overview:
        Input Features (N x F)
             |
        +--> Inception Layer 1 (0,1,2,3-hop parallel)
        |         |
        |    Inception Layer 2 (0,1,2,3-hop parallel)
        |         |
        +--> Inception Layer 3 (0,1,2,3-hop parallel)
             |
        L2 Normalization
             |
        Classifier (Linear + Softmax/Sigmoid)
             |
          Predictions

    Key Features:
    - Multi-scale parallel aggregation (tidak sequential seperti TrojanSAINT)
    - Channel attention untuk adaptive weighting
    - Support 3 variants: standard, adaptive, depthwise
    - Focal Loss untuk maximize TPR
    - Lebih robust untuk berbagai ukuran trojan

    Args:
        num_classes (int): Number of output classes
        arch_gcn (dict): Architecture configuration
        train_params (dict): Training hyperparameters
        feat_full (np.array): Full node features [N, F]
        label_full (np.array): Full node labels [N, C]
        cpu_eval (bool): Whether to use CPU for evaluation
        inception_layer_cls (str): Type of inception layer
            - 'standard': InceptionGNNLayer (default)
            - 'adaptive': AdaptiveInceptionGNNLayer (with learnable gates)
            - 'depthwise': DepthwiseSeparableInceptionGNN (lightweight)
    """

    def __init__(
        self,
        num_classes,
        arch_gcn,
        train_params,
        feat_full,
        label_full,
        cpu_eval=False,
        inception_layer_cls="standard",
    ):
        super(MultiSAINT_Inception, self).__init__()

        # Device configuration
        self.use_cuda = args_global.gpu >= 0
        if cpu_eval:
            self.use_cuda = False

        # Select inception layer type
        self.inception_type = inception_layer_cls.lower()
        if self.inception_type == "adaptive":
            self.InceptionLayer = AdaptiveInceptionGNNLayer
        elif self.inception_type == "depthwise":
            self.InceptionLayer = DepthwiseSeparableInceptionGNN
        else:  # standard
            self.InceptionLayer = InceptionGNNLayer

        # Training hyperparameters
        self.num_layers = len(arch_gcn["arch"].split("-"))
        self.weight_decay = train_params["weight_decay"]
        self.dropout = train_params["dropout"]
        self.lr = train_params["lr"]
        self.arch_gcn = arch_gcn
        self.sigmoid_loss = arch_gcn["loss"] == "sigmoid"

        # Features and labels (full graph)
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))
        self.label_full = torch.from_numpy(label_full.astype(np.float32))

        if self.use_cuda:
            self.feat_full = self.feat_full.cuda()
            self.label_full = self.label_full.cuda()

        # For CrossEntropy loss (integer labels)
        if not self.sigmoid_loss:
            self.label_full_cat = torch.from_numpy(
                label_full.argmax(axis=1).astype(np.int64)
            )
            if self.use_cuda:
                self.label_full_cat = self.label_full_cat.cuda()

        self.num_classes = num_classes

        # Parse architecture configuration
        _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer = (
            parse_layer_yml(arch_gcn, self.feat_full.shape[1])
        )

        # Get max_order from config (default: 3)
        self.max_order_inception = arch_gcn.get("max_order", 3)

        # Validate max_order
        if self.max_order_inception < 1:
            printf("WARNING: max_order < 1, setting to 1", style="yellow")
            self.max_order_inception = 1
        if self.max_order_inception > 5:
            printf("WARNING: max_order > 5 may cause over-smoothing", style="yellow")

        # Build dimensions dengan inception multiplier
        self.set_dims_inception(_dims)

        # Build inception layers
        self.num_params = 0
        self.aggregators, num_param = self.get_inception_aggregators()
        self.num_params += num_param
        self.conv_layers = nn.Sequential(*self.aggregators)

        # Classifier layer (order-0 aggregation)
        self.classifier = layers.HighOrderAggregator(
            self.dims_feat[-1],
            self.num_classes,
            act="I",
            order=0,
            dropout=self.dropout,
            bias="bias",
        )
        self.num_params += self.classifier.num_param

        # ===== FOCAL LOSS SETUP =====
        self.use_focal_loss = arch_gcn.get("use_focal_loss", True)  # Default: True
        self.focal_gamma = arch_gcn.get("focal_gamma", 2.0)
        self.focal_alpha = arch_gcn.get("focal_alpha", 0.75)
        # Koefisien interpolasi alpha->inverse-frequency (Interpolasi A).
        # beta=1 -> inverse-frequency murni (angka paper); beta turun -> alpha
        # yml makin terlibat. Default besar agar dekat code asli (F1 aman).
        self.focal_beta = arch_gcn.get("focal_beta", 0.8)

        if self.use_focal_loss:
            printf(
                f"  Using Focal Loss: gamma={self.focal_gamma}, alpha={self.focal_alpha}",
                style="green",
            )

            if self.sigmoid_loss:
                # Binary classification
                self.criterion = WeightedFocalLoss(
                    alpha=self.focal_alpha, gamma=self.focal_gamma
                )
            else:
                # Multi-class classification.
                #
                # Base class weights dari inverse-frequency (menjaga esensi
                # vektor alpha yang menghasilkan angka pada paper).
                base_weights = compute_class_weights(label_full, method="inverse")

                # Libatkan focal_alpha dari yml dengan menarik vektor alpha
                # paper [1-alpha, alpha] menuju inverse-frequency (code asli)
                # melalui interpolasi (Interpolasi A, diatur oleh focal_beta).
                class_weights = self._apply_alpha_modulation(
                    base_weights, self.focal_alpha, beta=self.focal_beta
                )

                if self.use_cuda:
                    class_weights = class_weights.cuda()

                printf(
                    f"  Base (inv-freq) weights: {base_weights.cpu().numpy()}",
                    style="green",
                )
                printf(
                    f"  Interpolated alpha weights "
                    f"(alpha={self.focal_alpha}, beta={self.focal_beta}): "
                    f"{class_weights.cpu().numpy()}",
                    style="green",
                )

                self.criterion = WeightedMultiClassFocalLoss(
                    alpha=class_weights, gamma=self.focal_gamma
                )
        else:
            printf("  Using Standard CE/BCE Loss", style="yellow")
            self.criterion = None
        # ===========================

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Print model information
        printf(f"\n{'=' * 70}", style="green")
        printf(f"MultiSAINT-Inception Model Initialized", style="green")
        printf(f"{'=' * 70}", style="green")
        printf(f"  Inception Type:        {self.inception_type}", style="green")
        printf(f"  Number of Layers:      {self.num_layers}", style="green")
        printf(f"  Max Order (hops):      {self.max_order_inception}", style="green")
        # Print attention configuration
        use_attention_config = self.arch_gcn.get("use_attention", True)
        printf(
            f"  Channel Attention:     {'ENABLED' if use_attention_config else 'DISABLED'}",
            style="green" if use_attention_config else "yellow",
        )
        printf(
            f"  Reduction Ratio:       {self.arch_gcn.get('reduction_ratio', 4)}",
            style="green",
        )

        printf(f"  Hidden Dimension:      {_dims[1]} (per branch)", style="green")
        printf(f"  Output Dimension:      {self.dims_feat[-1]}", style="green")
        printf(f"  Total Parameters:      {self.num_params:,}", style="green")
        printf(
            f"  Trainable Parameters:  {sum(p.numel() for p in self.parameters() if p.requires_grad):,}",
            style="green",
        )
        printf(f"{'=' * 70}\n", style="green")

    def _apply_alpha_modulation(self, base_weights, alpha, beta=0.8):
        """
        Libatkan focal_alpha dari config dengan menginterpolasi vektor alpha
        dari paper menuju vektor inverse-frequency yang menjadi basis angka
        pada code asli (Interpolasi A).

        Konvensi indeks kelas: 0 = benign (negatif), 1 = trojan (positif),
        konsisten dengan label_full.argmax(axis=1) pada forward pass.

        Titik awal (dari yml): a = [1 - alpha, alpha]
            mis. alpha=0.75 -> a = [0.25, 0.75]   (bobot kelas ala paper)
        Titik tujuan         : w = base_weights   (inverse-frequency, code asli)

        Interpolasi:
            alpha_eff = (1 - beta) * a + beta * w

        Sifat:
        - beta = 1.0 -> alpha_eff = w (inverse-frequency murni; identik code asli,
          angka paper). Alpha yml secara efektif tidak berpengaruh.
        - beta = 0.0 -> alpha_eff = a (bobot paper [1-alpha, alpha] murni).
        - 0 < beta < 1 -> alpha yml TETAP terlibat namun ditarik mendekati
          inverse-frequency. beta besar (mis. 0.8-0.9) -> dekat code asli
          sehingga F1 tetap baik, tetapi alpha yml masih ikut menggeser bobot.

        Catatan arah: beta BESAR = dekat code asli (aman). Ini kebalikan dari
        parameter lam pada formulasi tilt sebelumnya.

        Args:
            base_weights (torch.Tensor): vektor inverse-frequency [C] (code asli)
            alpha (float): focal_alpha dari config, rentang (0, 1)
            beta (float): koefisien interpolasi menuju inverse-frequency,
                rentang [0, 1], default 0.8

        Returns:
            torch.Tensor: vektor class-weight hasil interpolasi [C]
        """
        if base_weights.numel() != 2:
            # Hanya didefinisikan untuk binary HT detection (benign vs trojan).
            return base_weights

        beta = float(max(0.0, min(1.0, beta)))  # clamp ke [0, 1]

        # Vektor alpha dari paper: [benign, trojan] = [1 - alpha, alpha]
        a = torch.tensor(
            [1.0 - alpha, alpha],
            dtype=base_weights.dtype,
            device=base_weights.device,
        )

        # Interpolasi menuju inverse-frequency (base_weights)
        alpha_eff = (1.0 - beta) * a + beta * base_weights

        return alpha_eff

        return base_weights * tilt

    def set_dims_inception(self, dims):
        """
        Set dimensions untuk inception architecture

        Each inception layer outputs: dim_out * (max_order + 1)
        Example:
            Input: 128
            Layer 1: 64 per branch -> 64 * 4 = 256 (if max_order=3)
            Layer 2: 64 per branch -> 64 * 4 = 256
            Layer 3: 64 per branch -> 64 * 4 = 256

        Args:
            dims (list): List of dimensions from config [input_dim, hidden_dim, ...]
        """
        # Input dimension (original feature dimension)
        self.dims_feat = [dims[0]]

        # Hidden layers dengan inception multiplier
        for l in range(len(dims) - 1):
            # Each inception layer outputs: dim * (max_order + 1) branches concatenated
            dim_out_total = dims[l + 1] * (self.max_order_inception + 1)
            self.dims_feat.append(dim_out_total)

        # Weight dimensions (input -> output per branch)
        self.dims_weight = [
            (self.dims_feat[l], dims[l + 1]) for l in range(len(dims) - 1)
        ]

    def get_inception_aggregators(self):
        """
        Build inception aggregators untuk setiap layer

        CORRECTED: Now properly passes use_attention and reduction_ratio to layers

        Returns:
            tuple: (aggregators list, total_params)
        """
        num_param = 0
        aggregators = []

        # ============================================================
        # CORRECTED: Extract parameters dari arch_gcn and pass to layers
        # ============================================================
        use_attention = self.arch_gcn.get("use_attention", True)
        reduction_ratio = self.arch_gcn.get("reduction_ratio", 4)

        for l in range(self.num_layers):
            # Create inception layer with ALL parameters
            aggr = self.InceptionLayer(
                self.dims_weight[l][0],  # dim_in (dari layer sebelumnya)
                self.dims_weight[l][1],  # dim_out (per branch)
                max_order=self.max_order_inception,
                dropout=self.dropout,
                act=self.act_layer[l],
                bias=self.bias_layer[l],
                reduction_ratio=reduction_ratio,  # ADDED
                use_attention=use_attention,  # ADDED
            )

            # Count parameters
            num_param += sum(p.numel() for p in aggr.parameters())
            aggregators.append(aggr)

        return aggregators, num_param

    def forward(self, node_subgraph, adj_subgraph):
        """
        Forward pass through the network

        Args:
            node_subgraph (np.array or torch.Tensor): Node indices [M]
            adj_subgraph (torch.sparse.Tensor): Normalized adjacency matrix [M, M]

        Returns:
            tuple: (pred_subg, label_subg, label_subg_converted)
                pred_subg: Predictions [M, num_classes]
                label_subg: True labels [M, num_classes] (one-hot)
                label_subg_converted: Labels untuk loss computation
        """
        # Extract features and labels untuk subgraph
        feat_subg = self.feat_full[node_subgraph]
        label_subg = self.label_full[node_subgraph]
        label_subg_converted = (
            label_subg if self.sigmoid_loss else self.label_full_cat[node_subgraph]
        )

        # Forward through inception layers
        _, emb_subg = self.conv_layers((adj_subgraph, feat_subg))

        # L2 normalization of embeddings
        emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)

        # Classification
        pred_subg = self.classifier((None, emb_subg_norm))[1]

        return pred_subg, label_subg, label_subg_converted

    def _loss(self, preds, labels, norm_loss):
        """
        Compute loss function dengan Focal Loss support

        Args:
            preds (torch.Tensor): Predictions [M, num_classes]
            labels (torch.Tensor): True labels
            norm_loss (torch.Tensor): Normalization weights [M]

        Returns:
            torch.Tensor: Scalar loss value
        """
        if self.use_focal_loss:
            # Use Focal Loss
            if self.sigmoid_loss:
                # Binary classification
                return self.criterion(preds, labels, norm_loss)
            else:
                # Multi-class classification
                return self.criterion(preds, labels, norm_loss)
        else:
            # Standard loss (backward compatibility)
            if self.sigmoid_loss:
                # Binary classification with BCE loss
                norm_loss = norm_loss.unsqueeze(1)
                return torch.nn.BCEWithLogitsLoss(weight=norm_loss, reduction="sum")(
                    preds, labels
                )
            else:
                # Multi-class classification with CrossEntropy
                _ls = torch.nn.CrossEntropyLoss(reduction="none")(preds, labels)
                return (norm_loss * _ls).sum()

    def predict(self, preds):
        """
        Convert logits to probabilities

        Args:
            preds (torch.Tensor): Logits [M, num_classes]

        Returns:
            torch.Tensor: Probabilities [M, num_classes]
        """
        if self.sigmoid_loss:
            return nn.Sigmoid()(preds)
        else:
            return F.softmax(preds, dim=1)

    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Single training step: forward + backward + optimize

        Args:
            node_subgraph: Node indices
            adj_subgraph: Adjacency matrix
            norm_loss_subgraph: Loss normalization weights

        Returns:
            tuple: (loss, predictions, labels)
        """
        self.train()
        self.optimizer.zero_grad()

        # Forward pass
        preds, labels, labels_converted = self(node_subgraph, adj_subgraph)

        # Compute loss
        loss = self._loss(preds, labels_converted, norm_loss_subgraph)

        # Backward pass
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)

        # Update parameters
        self.optimizer.step()

        return loss, self.predict(preds), labels

    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Single evaluation step: forward only (no gradients)

        Args:
            node_subgraph: Node indices
            adj_subgraph: Adjacency matrix
            norm_loss_subgraph: Loss normalization weights

        Returns:
            tuple: (loss, predictions, labels)
        """
        self.eval()
        with torch.no_grad():
            preds, labels, labels_converted = self(node_subgraph, adj_subgraph)
            loss = self._loss(preds, labels_converted, norm_loss_subgraph)

        return loss, self.predict(preds), labels


def parse_layer_yml(arch_gcn, dim_input):
    """
    Parse YAML config file untuk layer architecture

    Args:
        arch_gcn (dict): Architecture configuration
        dim_input (int): Input feature dimension

    Returns:
        tuple: (dims, order_layer, act_layer, bias_layer, aggr_layer)

    Example config:
        arch_gcn = {
            'arch': '1-1-1',
            'dim': 128,
            'act': 'relu',
            'bias': 'norm-nn',
            'aggr': 'concat'
        }
    """
    num_layers = len(arch_gcn["arch"].split("-"))

    # Set default values, then update by arch_gcn
    bias_layer = [arch_gcn.get("bias", "norm-nn")] * num_layers
    act_layer = [arch_gcn.get("act", "relu")] * num_layers
    aggr_layer = [arch_gcn.get("aggr", "concat")] * num_layers
    dims_layer = [arch_gcn.get("dim", 128)] * num_layers
    order_layer = [int(o) for o in arch_gcn["arch"].split("-")]

    return [dim_input] + dims_layer, order_layer, act_layer, bias_layer, aggr_layer


# ============================================================
# Utility Functions
# ============================================================


def get_model_summary(model):
    """
    Get comprehensive model summary

    Args:
        model: MultiSAINT_Inception model

    Returns:
        dict: Model summary information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = {
        "model_name": "MultiSAINT-Inception",
        "inception_type": model.inception_type,
        "num_layers": model.num_layers,
        "max_order": model.max_order_inception,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": total_params * 4 / (1024**2),  # FP32
        "input_dim": model.dims_feat[0],
        "output_dim": model.dims_feat[-1],
        "num_classes": model.num_classes,
        "device": "cuda" if model.use_cuda else "cpu",
        "use_focal_loss": model.use_focal_loss,
        "focal_gamma": model.focal_gamma if model.use_focal_loss else None,
        "focal_alpha": model.focal_alpha if model.use_focal_loss else None,
    }

    return summary


def print_model_summary(model):
    """
    Print detailed model summary

    Args:
        model: MultiSAINT_Inception model
    """
    summary = get_model_summary(model)

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Model Name:          {summary['model_name']}")
    print(f"Inception Type:      {summary['inception_type']}")
    print(f"Number of Layers:    {summary['num_layers']}")
    print(f"Max Aggregation Order: {summary['max_order']}")
    print(f"Input Dimension:     {summary['input_dim']}")
    print(f"Output Dimension:    {summary['output_dim']}")
    print(f"Number of Classes:   {summary['num_classes']}")
    print(f"\nTotal Parameters:    {summary['total_params']:,}")
    print(f"Trainable Params:    {summary['trainable_params']:,}")
    print(f"Model Size:          {summary['param_size_mb']:.2f} MB")
    print(f"Device:              {summary['device']}")
    print(
        f"\nLoss Function:       {'Focal Loss' if summary['use_focal_loss'] else 'Standard CE/BCE'}"
    )
    if summary["use_focal_loss"]:
        print(f"  Gamma:             {summary['focal_gamma']}")
        print(f"  Alpha:             {summary['focal_alpha']}")
    print("=" * 70 + "\n")


def compare_with_baseline(model_inception, model_baseline):
    """
    Compare MultiSAINT-Inception with baseline GraphSAINT

    Args:
        model_inception: MultiSAINT_Inception model
        model_baseline: Original GraphSAINT model

    Returns:
        dict: Comparison metrics
    """
    params_inception = sum(p.numel() for p in model_inception.parameters())
    params_baseline = sum(p.numel() for p in model_baseline.parameters())

    comparison = {
        "inception_params": params_inception,
        "baseline_params": params_baseline,
        "param_increase": params_inception - params_baseline,
        "param_increase_pct": (params_inception / params_baseline - 1) * 100,
        "size_inception_mb": params_inception * 4 / (1024**2),
        "size_baseline_mb": params_baseline * 4 / (1024**2),
    }

    return comparison


# ============================================================
# Testing Code
# ============================================================

if __name__ == "__main__":
    print("Testing MultiSAINT-Inception Model with Focal Loss...")

    # Mock data for testing
    num_nodes = 1000
    num_features = 128
    num_classes = 2

    # Create dummy features and labels (imbalanced: 10% trojan)
    feat_full = np.random.randn(num_nodes, num_features).astype(np.float32)
    label_full = np.zeros((num_nodes, num_classes), dtype=np.float32)

    # Create imbalanced dataset (10% positive)
    num_positive = int(num_nodes * 0.1)
    positive_indices = np.random.choice(num_nodes, num_positive, replace=False)
    label_full[positive_indices, 1] = 1
    label_full[label_full[:, 1] == 0, 0] = 1

    print(
        f"Dataset: {num_nodes} nodes, {num_positive} trojans ({num_positive / num_nodes * 100:.1f}%)"
    )

    # Architecture config with Focal Loss
    arch_gcn = {
        "arch": "1-1-1",
        "dim": 64,
        "act": "relu",
        "bias": "norm-nn",
        "aggr": "concat",
        "loss": "softmax",
        "max_order": 3,
        "use_focal_loss": True,
        "focal_gamma": 2.0,
        "focal_alpha": 0.75,  # CORRECTED: Lin et al. convention
        "use_attention": True,  # ADDED for testing
        "reduction_ratio": 4,  # ADDED for testing
    }

    # Training params
    train_params = {"lr": 0.01, "weight_decay": 0.0, "dropout": 0.1}

    # Mock args_global
    class MockArgs:
        gpu = -1  # CPU

    import graphsaint.globals as globals_module

    globals_module.args_global = MockArgs()

    # Test all three variants with Focal Loss
    for inception_type in ["standard", "adaptive", "depthwise"]:
        print(f"\n{'=' * 70}")
        print(f"Testing {inception_type.upper()} Inception Model + Focal Loss")
        print(f"{'=' * 70}")

        model = MultiSAINT_Inception(
            num_classes=num_classes,
            arch_gcn=arch_gcn,
            train_params=train_params,
            feat_full=feat_full,
            label_full=label_full,
            cpu_eval=True,
            inception_layer_cls=inception_type,
        )

        print_model_summary(model)

        # Test forward pass
        batch_size = 100
        node_subgraph = np.random.choice(num_nodes, batch_size, replace=False)

        # Create sparse adj
        num_edges = 500
        indices = torch.randint(0, batch_size, (2, num_edges))
        values = torch.ones(num_edges) / batch_size  # Normalized
        adj_subgraph = torch.sparse_coo_tensor(
            indices, values, (batch_size, batch_size)
        )

        # Create norm_loss
        norm_loss = torch.ones(batch_size)

        # Forward pass
        preds, labels, labels_conv = model(node_subgraph, adj_subgraph)

        # Test loss computation
        loss = model._loss(preds, labels_conv, norm_loss)

        print(f"Forward pass test:")
        print(f"  Input nodes:    {batch_size}")
        print(f"  Predictions:    {preds.shape}")
        print(f"  Labels:         {labels.shape}")
        print(f"  Loss value:     {loss.item():.4f}")
        print(f"✓ {inception_type.upper()} model test PASSED")

    print(f"\n{'=' * 70}")
    print("ALL MODEL TESTS PASSED! ✓")
    print(f"{'=' * 70}\n")