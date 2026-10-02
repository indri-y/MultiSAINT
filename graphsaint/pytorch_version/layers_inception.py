"""
layers_inception.py - FULLY CORRECTED VERSION
MultiSAINT: Inception-style Graph Neural Network Layers
Multi-scale feature aggregation untuk Hardware Trojan Detection di FPGA GLN

CORRECTIONS APPLIED:
1. [FIXED] InceptionGNNLayer - use_attention parameter works correctly
2. [FIXED] AdaptiveInceptionGNNLayer - ADDED use_attention parameter (was always ON)
3. [FIXED] DepthwiseSeparableInceptionGNN - ADDED use_attention parameter (was missing)

Author: MultiSAINT Research Team
Purpose: Doctoral Research - FPGA Hardware Trojan Detection
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# Activation functions dictionary
F_ACT = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(0.2),
    "elu": nn.ELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "I": lambda x: x,
}


class InceptionGNNLayer(nn.Module):
    """
    Standard Inception-GNN Layer dengan multi-scale parallel aggregation

    CORRECTED: use_attention parameter properly controls channel attention

    Args:
        dim_in (int): Input feature dimension
        dim_out (int): Output feature dimension per branch
        max_order (int): Maximum hop untuk aggregation (0 to max_order)
        dropout (float): Dropout rate
        act (str): Activation function name
        bias (str): Bias type ('bias', 'norm', 'norm-nn')
        reduction_ratio (int): Reduction ratio untuk channel attention
        use_attention (bool): Whether to use channel attention (SE module)

    Output dimension: dim_out * (max_order + 1)
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        max_order=3,
        dropout=0.0,
        act="relu",
        bias="norm-nn",
        reduction_ratio=4,
        use_attention=True,
        **kwargs,
    ):
        super(InceptionGNNLayer, self).__init__()

        assert bias in ["bias", "norm", "norm-nn"], f"Unknown bias type: {bias}"
        assert act in F_ACT, f"Unknown activation: {act}"

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.max_order = max_order
        self.act_fn = F_ACT[act]
        self.bias = bias
        self.dropout = dropout
        self.use_attention = use_attention

        # Multi-scale branches
        self.branches = nn.ModuleList()
        for order in range(self.max_order + 1):
            if bias == "norm-nn":
                branch = nn.Sequential(
                    nn.Linear(dim_in, dim_out, bias=True),
                    nn.BatchNorm1d(dim_out),
                )
            else:
                branch = nn.Linear(dim_in, dim_out, bias=True)

            nn.init.xavier_uniform_(
                branch[0].weight if bias == "norm-nn" else branch.weight
            )
            self.branches.append(branch)

        # Channel Attention Module (conditional)
        total_dim = dim_out * (self.max_order + 1)

        if self.use_attention:
            hidden_dim = max(total_dim // reduction_ratio, 16)

            self.channel_attention = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, total_dim),
                nn.Sigmoid(),
            )
        else:
            self.channel_attention = None

        # Dropout layer
        self.f_dropout = nn.Dropout(p=self.dropout)

        # Count parameters
        self.num_param = sum(p.numel() for p in self.parameters())

    def _spmm(self, adj_norm, feat):
        """Sparse matrix multiplication"""
        return torch.sparse.mm(adj_norm, feat)

    def forward(self, inputs):
        """
        Forward pass dengan multi-scale parallel aggregation

        Args:
            inputs: tuple (adj_norm, feat_in)

        Returns:
            tuple (adj_norm, feat_out)
        """
        adj_norm, feat_in = inputs
        feat_in = self.f_dropout(feat_in)

        # Generate multi-hop features
        feat_hops = [feat_in]
        for order in range(1, self.max_order + 1):
            feat_hop = self._spmm(adj_norm, feat_hops[-1])
            feat_hops.append(feat_hop)

        # Process each hop with separate branch
        branch_outputs = []
        for order, feat_hop in enumerate(feat_hops):
            if self.bias == "norm-nn":
                branch_out = self.branches[order](feat_hop)
            else:
                branch_out = self.branches[order](feat_hop)

            branch_out = self.act_fn(branch_out)
            branch_outputs.append(branch_out)

        # Concatenate all branches
        feat_concat = torch.cat(branch_outputs, dim=1)

        # Channel attention (conditional) - CORRECTED
        if self.use_attention and self.channel_attention is not None:
            feat_gap = feat_concat.mean(dim=0, keepdim=True)
            attention_weights = self.channel_attention(feat_gap)
            feat_out = feat_concat * attention_weights
        else:
            # No attention - pass through directly
            feat_out = feat_concat

        return adj_norm, feat_out


class AdaptiveInceptionGNNLayer(nn.Module):
    """
    Adaptive Inception-GNN dengan learnable fusion gates

    CORRECTED: Added use_attention parameter (was always ON before)

    Args:
        dim_in (int): Input feature dimension
        dim_out (int): Output feature dimension per branch
        max_order (int): Maximum hop untuk aggregation
        dropout (float): Dropout rate
        act (str): Activation function name
        bias (str): Bias type
        use_residual (bool): Whether to use residual connection
        reduction_ratio (int): Reduction ratio untuk channel attention
        use_attention (bool): Whether to use channel attention
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        max_order=3,
        dropout=0.0,
        act="relu",
        bias="norm-nn",
        use_residual=False,
        reduction_ratio=4,
        use_attention=True,
        **kwargs,
    ):
        super(AdaptiveInceptionGNNLayer, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.max_order = max_order
        self.act_fn = F_ACT[act]
        self.bias = bias
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_attention = use_attention  # ADDED

        # Multi-scale branches
        self.branches = nn.ModuleList()
        for order in range(self.max_order + 1):
            if bias == "norm-nn":
                branch = nn.Sequential(
                    nn.Linear(dim_in, dim_out, bias=True),
                    nn.BatchNorm1d(dim_out),
                )
            else:
                branch = nn.Linear(dim_in, dim_out, bias=True)

            nn.init.xavier_uniform_(
                branch[0].weight if bias == "norm-nn" else branch.weight
            )
            self.branches.append(branch)

        # Learnable fusion gates
        self.fusion_gates = nn.Parameter(torch.ones(self.max_order + 1))

        # Residual projection
        total_dim_out = dim_out * (self.max_order + 1)
        if use_residual and dim_in != total_dim_out:
            self.residual_projection = nn.Linear(dim_in, total_dim_out, bias=False)
        else:
            self.residual_projection = None

        # Channel attention (conditional) - CORRECTED
        if self.use_attention:
            hidden_dim = max(total_dim_out // reduction_ratio, 16)
            self.channel_attention = nn.Sequential(
                nn.Linear(total_dim_out, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, total_dim_out),
                nn.Sigmoid(),
            )
        else:
            self.channel_attention = None

        self.f_dropout = nn.Dropout(p=self.dropout)
        self.num_param = sum(p.numel() for p in self.parameters())

    def _spmm(self, adj_norm, feat):
        return torch.sparse.mm(adj_norm, feat)

    def forward(self, inputs):
        adj_norm, feat_in = inputs
        feat_in_original = feat_in
        feat_in = self.f_dropout(feat_in)

        # Multi-hop features
        feat_hops = [feat_in]
        for order in range(1, self.max_order + 1):
            feat_hops.append(self._spmm(adj_norm, feat_hops[-1]))

        # Process with fusion gates
        branch_outputs = []
        fusion_weights = F.softmax(self.fusion_gates, dim=0)

        for order, feat_hop in enumerate(feat_hops):
            if self.bias == "norm-nn":
                branch_out = self.branches[order](feat_hop)
            else:
                branch_out = self.branches[order](feat_hop)

            branch_out = self.act_fn(branch_out)
            branch_out = branch_out * fusion_weights[order]
            branch_outputs.append(branch_out)

        feat_concat = torch.cat(branch_outputs, dim=1)

        # Channel attention (conditional) - CORRECTED
        if self.use_attention and self.channel_attention is not None:
            feat_gap = feat_concat.mean(dim=0, keepdim=True)
            attention_weights = self.channel_attention(feat_gap)
            feat_out = feat_concat * attention_weights
        else:
            # No attention - pass through directly
            feat_out = feat_concat

        # Residual connection
        if self.use_residual:
            if self.residual_projection is not None:
                feat_out = feat_out + self.residual_projection(feat_in_original)
            else:
                feat_out = feat_out + feat_in_original

        return adj_norm, feat_out


class DepthwiseSeparableInceptionGNN(nn.Module):
    """
    Depthwise Separable Inception-GNN (Lightweight Variant)

    CORRECTED: Added use_attention parameter (was missing before)

    Args:
        dim_in (int): Input feature dimension
        dim_out (int): Output feature dimension per branch
        max_order (int): Maximum hop untuk aggregation
        dropout (float): Dropout rate
        act (str): Activation function name
        bias (str): Bias type
        reduction_ratio (int): Reduction ratio untuk channel attention
        use_attention (bool): Whether to use channel attention
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        max_order=3,
        dropout=0.0,
        act="relu",
        bias="norm-nn",
        reduction_ratio=4,
        use_attention=True,
        **kwargs,
    ):
        super(DepthwiseSeparableInceptionGNN, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.max_order = max_order
        self.act_fn = F_ACT[act]
        self.dropout = dropout
        self.bias = bias
        self.use_attention = use_attention  # ADDED

        # Depthwise branches
        self.depthwise_branches = nn.ModuleList()
        for order in range(self.max_order + 1):
            depthwise = nn.Linear(dim_in, dim_in, bias=False)
            nn.init.xavier_uniform_(depthwise.weight)
            self.depthwise_branches.append(depthwise)

        # Pointwise layer
        total_dim_intermediate = dim_in * (self.max_order + 1)
        total_dim_output = dim_out * (self.max_order + 1)

        if bias == "norm-nn":
            self.pointwise = nn.Sequential(
                nn.Linear(total_dim_intermediate, total_dim_output, bias=True),
                nn.BatchNorm1d(total_dim_output),
            )
        else:
            self.pointwise = nn.Linear(
                total_dim_intermediate, total_dim_output, bias=True
            )

        nn.init.xavier_uniform_(
            self.pointwise[0].weight if bias == "norm-nn" else self.pointwise.weight
        )

        # Channel attention (conditional) - ADDED
        if self.use_attention:
            hidden_dim = max(total_dim_output // reduction_ratio, 16)
            self.channel_attention = nn.Sequential(
                nn.Linear(total_dim_output, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, total_dim_output),
                nn.Sigmoid(),
            )
        else:
            self.channel_attention = None

        self.f_dropout = nn.Dropout(p=self.dropout)
        self.num_param = sum(p.numel() for p in self.parameters())

    def _spmm(self, adj_norm, feat):
        return torch.sparse.mm(adj_norm, feat)

    def forward(self, inputs):
        adj_norm, feat_in = inputs
        feat_in = self.f_dropout(feat_in)

        # Multi-hop features
        feat_hops = [feat_in]
        for order in range(1, self.max_order + 1):
            feat_hops.append(self._spmm(adj_norm, feat_hops[-1]))

        # Depthwise processing
        branch_outputs = []
        for order, feat_hop in enumerate(feat_hops):
            branch_out = self.depthwise_branches[order](feat_hop)
            branch_outputs.append(branch_out)

        # Concatenate and pointwise
        feat_concat = torch.cat(branch_outputs, dim=1)

        if self.bias == "norm-nn":
            feat_out = self.pointwise(feat_concat)
        else:
            feat_out = self.pointwise(feat_concat)

        feat_out = self.act_fn(feat_out)

        # Channel attention (conditional) - ADDED
        if self.use_attention and self.channel_attention is not None:
            feat_gap = feat_out.mean(dim=0, keepdim=True)
            attention_weights = self.channel_attention(feat_gap)
            feat_out = feat_out * attention_weights

        return adj_norm, feat_out


# ============================================================
# VERIFICATION CODE
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING CORRECTED LAYERS - use_attention ON/OFF")
    print("=" * 70)

    # Test parameters
    batch_size = 100
    dim_in = 128
    dim_out = 64
    max_order = 3

    # Create dummy data
    feat = torch.randn(batch_size, dim_in)
    indices = torch.randint(0, batch_size, (2, 500))
    values = torch.ones(500) / batch_size
    adj = torch.sparse_coo_tensor(indices, values, (batch_size, batch_size))

    # Test all three layer types with attention ON and OFF
    layer_classes = [
        ("InceptionGNNLayer", InceptionGNNLayer),
        ("AdaptiveInceptionGNNLayer", AdaptiveInceptionGNNLayer),
        ("DepthwiseSeparableInceptionGNN", DepthwiseSeparableInceptionGNN),
    ]

    for layer_name, LayerClass in layer_classes:
        print(f"\n{'=' * 50}")
        print(f"Testing: {layer_name}")
        print("=" * 50)

        for use_att in [True, False]:
            layer = LayerClass(
                dim_in=dim_in,
                dim_out=dim_out,
                max_order=max_order,
                use_attention=use_att,
                reduction_ratio=4,
            )

            # Check if attention module exists
            has_attention = (
                hasattr(layer, "channel_attention")
                and layer.channel_attention is not None
            )

            # Forward pass
            _, out = layer((adj, feat))

            param_count = sum(p.numel() for p in layer.parameters())

            print(f"  use_attention={use_att}:")
            print(f"    - Attention module exists: {has_attention}")
            print(f"    - Output shape: {out.shape}")
            print(f"    - Parameters: {param_count:,}")

            # Verify attention is actually OFF when use_attention=False
            if not use_att:
                assert not has_attention, (
                    f"ERROR: Attention should be None when use_attention=False!"
                )
                print(f"    ✅ Verified: No attention module when use_attention=False")
            else:
                assert has_attention, (
                    f"ERROR: Attention should exist when use_attention=True!"
                )
                print(
                    f"    ✅ Verified: Attention module exists when use_attention=True"
                )

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - use_attention works correctly!")
    print("=" * 70)
