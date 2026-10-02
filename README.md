# MultiSAINT: Parallel Multi-Scale GNN for FPGA Hardware Trojan Detection

Official implementation of the paper published in *IEEE Access*, vol. 14, 2026.
DOI: [10.1109/ACCESS.2026.3689539](https://doi.org/10.1109/ACCESS.2026.3689539)

## Installation

Tested with Python 3.12 and CUDA 12.4.

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python graphsaint/setup.py build_ext --inplace
```

## Dataset

The dataset is built from Trust-Hub RTL benchmarks synthesized with Xilinx Vivado 2023.1 (xc7z020clg400-1). The netlist-to-graph parser is provided in `dataset_prep/`. The data folder follows the GraphSAINT format (`adj_full.npz`, `adj_train.npz`, `feats.npy`, `class_map.json`, `role.json`). The processed dataset is available from the authors upon request.

## Training

```bash
python -m graphsaint.pytorch_version.train_inception \
    --data_prefix ./data/<dataset> \
    --train_config train_configs/multisaint_order3_noattention.yml \
    --seed 42 \
    --gpu 0
```

| Config file | Paper |
|---|---|
| `multisaint_order3_noattention.yml` | Config-6 (main model) |
| `multisaint_order3_attention.yml` | Config-5 |

## Trained Models

Checkpoints for Config-6 (5 seeds) are available under [Releases](https://github.com/indri-y/MultiSAINT/releases).

## Notes

- Focal Loss class weights are computed from inverse class frequency (`focal_beta: 1.0`), which is the setting used to produce the reported results.
- Results may vary slightly between runs due to non-deterministic GPU operations.
- Seeds used in the paper: 42, 123, 456, 789, 1024.

## Acknowledgement

This code is built on [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) (Zeng et al., ICLR 2020), through the version adapted for [TrojanSAINT](https://github.com/DfX-NYUAD/TrojanSAINT) by DfX-NYUAD. The original GraphSAINT license is included in `LICENSE-GraphSAINT`.

## Citation

```bibtex
@article{yanti2026multisaint,
  author  = {Yanti, Indri and Istiyanto, Jazi Eko and Natan, Oskar},
  title   = {MultiSAINT: Parallel Multi-Scale GNN for FPGA Hardware Trojan Detection},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {68166--68185},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3689539}
}
```
