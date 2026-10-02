# minibatch.py
from graphsaint.globals import *
import math
from graphsaint.utils import *
from graphsaint.graph_samplers import *
from graphsaint.norm_aggr import *
import torch
import scipy.sparse as sp
import numpy as np
import time


def _coo_scipy2torch(adj):
    """
    Convert scipy.sparse matrix (any) to torch.sparse_coo_tensor (float32).
    """
    coo = adj if sp.isspmatrix_coo(adj) else adj.tocoo()
    indices = np.vstack((coo.row, coo.col)).astype(np.int64)
    i = torch.from_numpy(indices)
    v = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, size=coo.shape)


class Minibatch:
    """
    Menyediakan minibatch untuk trainer/evaluator.
    """

    def __init__(self, adj_full_norm, adj_train, role, train_params, cpu_eval=False):
        """
        Inputs:
            adj_full_norm   scipy CSR, adj normalisasi (row-normalized) graf penuh
            adj_train       scipy CSR, adj graf training (hanya edge train-train)
            role            dict: 'tr','va','te' -> list indeks node
            train_params    dict: parameter training & sampling
            cpu_eval        bool: eval full-batch di CPU (hindari OOM)
        """
        self.use_cuda = args_global.gpu >= 0
        if cpu_eval:
            self.use_cuda = False

        # role arrays
        self.node_train = np.array(role["tr"], dtype=np.int64)
        self.node_val = np.array(role["va"], dtype=np.int64)
        self.node_test = np.array(role["te"], dtype=np.int64)

        # simpan graf (torch sparse untuk full graph)
        self.adj_full_norm = _coo_scipy2torch(adj_full_norm)
        self.adj_train = adj_train.tocsr().astype(np.float32)
        if self.use_cuda:
            self.adj_full_norm = self.adj_full_norm.cuda()

        # derajat untuk normalisasi subgraf nantinya
        self.deg_train = np.asarray(self.adj_train.sum(1)).ravel().astype(np.float32)

        # Optional: prune node training deg=0 (hemat sampling)
        _mask_noniso = self.deg_train[self.node_train] > 0
        if _mask_noniso.sum() < self.node_train.size:
            printf(
                f"Pruning {self.node_train.size - _mask_noniso.sum()} isolated training nodes (deg=0).",
                style="yellow",
            )
            self.node_train = self.node_train[_mask_noniso]

        # book-keeping minibatch
        self.node_subgraph = None
        self.batch_num = -1

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []

        # loss normalization (train & test/full)
        self.norm_loss_train = np.zeros(self.adj_train.shape[0], dtype=np.float32)
        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0], dtype=np.float32)

        # Full-batch eval: bobot rata untuk semua node agar konsisten
        _denom = len(self.node_train) + len(self.node_val) + len(self.node_test)
        if _denom == 0:
            _denom = self.adj_full_norm.shape[0]
        val_per_node = 1.0 / float(_denom)
        self.norm_loss_test[self.node_train] = val_per_node
        self.norm_loss_test[self.node_val] = val_per_node
        self.norm_loss_test[self.node_test] = val_per_node
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_test = self.norm_loss_test.cuda()

        self.norm_aggr_train = np.zeros(self.adj_train.size, dtype=np.float32)

        # parameter sampling
        self.sample_coverage = train_params["sample_coverage"]

    def set_sampler(self, train_phases):
        """
        Pilih sampler dan lakukan warm-up untuk estimasi faktor normalisasi.
        """
        # reset buffer
        self.subgraphs_remaining_indptr.clear()
        self.subgraphs_remaining_indices.clear()
        self.subgraphs_remaining_data.clear()
        self.subgraphs_remaining_nodes.clear()
        self.subgraphs_remaining_edge_index.clear()

        self.method_sample = train_phases["sampler"]

        if self.method_sample == "mrw":
            _deg_clip = int(train_phases.get("deg_clip", 100000))
            self.size_subg_budget = int(train_phases["size_subgraph"])
            self.graph_sampler = mrw_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                int(train_phases["size_frontier"]),
                _deg_clip,
            )
        elif self.method_sample == "rw":
            self.size_subg_budget = int(train_phases["num_root"]) * int(
                train_phases["depth"]
            )
            self.graph_sampler = rw_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                int(train_phases["num_root"]),
                int(train_phases["depth"]),
            )
        elif self.method_sample == "edge":
            self.size_subg_budget = int(train_phases["size_subg_edge"]) * 2
            self.graph_sampler = edge_sampling(
                self.adj_train,
                self.node_train,
                int(train_phases["size_subg_edge"]),
            )
        elif self.method_sample == "node":
            self.size_subg_budget = int(train_phases["size_subgraph"])
            self.graph_sampler = node_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        elif self.method_sample == "full_batch":
            self.size_subg_budget = self.node_train.size
            self.graph_sampler = full_batch_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        elif self.method_sample == "vanilla_node_python":
            self.size_subg_budget = int(train_phases["size_subgraph"])
            self.graph_sampler = NodeSamplingVanillaPython(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        else:
            raise NotImplementedError(f"Unknown sampler: {self.method_sample}")

        # Estimasi faktor normalisasi (alpha/lambda) via warm-up sampling
        self.norm_loss_train = np.zeros(self.adj_train.shape[0], dtype=np.float32)
        self.norm_aggr_train = np.zeros(self.adj_train.size, dtype=np.float32)

        tot_sampled_nodes = 0
        while True:
            self.par_graph_sample("train")
            tot_sampled_nodes = sum(len(n) for n in self.subgraphs_remaining_nodes)
            if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
                break
        print()

        num_subg = len(self.subgraphs_remaining_nodes)
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1.0
            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1.0

        # seharusnya val/test tidak ter-sample
        assert (
            self.norm_loss_train[self.node_val].sum()
            + self.norm_loss_train[self.node_test].sum()
            == 0
        )

        # Hitung faktor agregasi per-edge (hindari NaN/inf)
        for v in range(self.adj_train.shape[0]):
            i_s = self.adj_train.indptr[v]
            i_e = self.adj_train.indptr[v + 1]
            if i_e > i_s:
                with np.errstate(divide="ignore", invalid="ignore"):
                    val = self.norm_loss_train[v] / self.norm_aggr_train[i_s:i_e]
                val = np.clip(val, 0.0, 1e4)
                val[~np.isfinite(val)] = 0.1
                self.norm_aggr_train[i_s:i_e] = val

        # node yang tidak pernah ter-sample -> set kecil
        self.norm_loss_train[self.norm_loss_train == 0] = 0.1
        self.norm_loss_train[self.node_val] = 0.0
        self.norm_loss_train[self.node_test] = 0.0
        # normalisasi ke jumlah subgraf
        self.norm_loss_train[self.node_train] = (
            num_subg / self.norm_loss_train[self.node_train] / self.node_train.size
        )
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_train = self.norm_loss_train.cuda()

    def par_graph_sample(self, phase):
        """
        Jalankan sampling paralel (wrapper Cython).
        """
        t0 = time.time()
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(phase)
        t1 = time.time()
        print("sampling 200 subgraphs:   time = {:.3f} sec".format(t1 - t0), end="\r")
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)

    def one_batch(self, mode="train"):
        """
        Buat satu batch:
          - train : satu subgraf hasil sampler
          - val/test/valtest : full-batch pada graf penuh
        Return:
          node_subgraph (np.array), adj (torch sparse atau scipy CSR), norm_loss (torch tensor)
        """
        if mode in ["val", "test", "valtest"]:
            self.node_subgraph = np.arange(self.adj_full_norm.shape[0], dtype=np.int64)
            adj = self.adj_full_norm
        else:
            assert mode == "train"
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample("train")
                print()

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)
            adj = sp.csr_matrix(
                (
                    self.subgraphs_remaining_data.pop(),
                    self.subgraphs_remaining_indices.pop(),
                    self.subgraphs_remaining_indptr.pop(),
                ),
                shape=(self.size_subgraph, self.size_subgraph),
                dtype=np.float32,
            )
            adj_edge_index = self.subgraphs_remaining_edge_index.pop()
            # Normalisasi faktor agregasi per-edge
            norm_aggr(
                adj.data,
                adj_edge_index,
                self.norm_aggr_train,
                num_proc=args_global.num_cpu_core,
            )

            # Row-normalize subgraf pakai derajat node asli yang ter-ambil
            adj = adj_norm(adj, deg=self.deg_train[self.node_subgraph])
            # ke torch sparse
            adj = _coo_scipy2torch(adj)
            if self.use_cuda:
                adj = adj.cuda()

            self.batch_num += 1

        norm_loss = (
            self.norm_loss_test
            if mode in ["val", "test", "valtest"]
            else self.norm_loss_train
        )
        norm_loss = norm_loss[self.node_subgraph]
        return self.node_subgraph, adj, norm_loss

    def num_training_batches(self):
        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))

    def shuffle(self):
        self.node_train = np.random.permutation(self.node_train)
        self.batch_num = -1

    def end(self):
        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]
