from graphsaint.globals import *
import numpy as np
import scipy.sparse
import time
import math
from math import ceil
import graphsaint.cython_sampler as cy


class GraphSampler:
    """
    Super-class untuk semua sampler GraphSAINT.
    """

    def __init__(self, adj_train, node_train, size_subgraph, args_preproc):
        """
        Inputs:
            adj_train       scipy CSR, graf training (N x N)
            node_train      1D np.array indeks node training
            size_subgraph   estimasi jumlah node per subgraf
            args_preproc    argumen tambahan untuk pre-processing
        """
        # Pastikan CSR float32
        self.adj_train = adj_train.tocsr().astype(np.float32)
        self.node_train = np.unique(np.asarray(node_train, dtype=np.int32))
        self.size_subgraph = int(size_subgraph)
        self.name_sampler = "None"
        self.node_subgraph = None

        # tempat buffer untuk hasil par_sample (diisi oleh cython)
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []

        # pre-processing spesifik sampler
        self.preproc(**args_preproc)

    def preproc(self, **kwargs):
        pass

    def par_sample(self, stage, **kwargs):
        """
        Panggil sampler paralel di Cython; mengembalikan
        list of (indptr, indices, data, nodes, edge_index) untuk beberapa subgraf.
        """
        return self.cy_sampler.par_sample()

    def _helper_extract_subgraph(self, node_ids):
        """
        Hanya untuk contoh sampler Python murni (bukan Cython).
        Return adj subgraf (CSR) + mapping node/edge ke graf asli.
        """
        node_ids = np.unique(np.asarray(node_ids, dtype=np.int64))
        node_ids.sort()
        orig2subg = {n: i for i, n in enumerate(node_ids)}
        n = node_ids.size
        indptr = np.zeros(n + 1, dtype=np.int64)
        indices = []
        subg_edge_index = []
        subg_nodes = node_ids
        for nid in node_ids:
            idx_s, idx_e = self.adj_train.indptr[nid], self.adj_train.indptr[nid + 1]
            neighs = self.adj_train.indices[idx_s:idx_e]
            for i_n, nb in enumerate(neighs):
                if nb in orig2subg:
                    indices.append(orig2subg[nb])
                    indptr[orig2subg[nid] + 1] += 1
                    subg_edge_index.append(idx_s + i_n)
        indptr = indptr.cumsum().astype(np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        subg_edge_index = np.asarray(subg_edge_index, dtype=np.int64)
        data = np.ones(indices.size, dtype=np.float32)
        assert indptr[-1] == indices.size == subg_edge_index.size
        return indptr, indices, data, subg_nodes, subg_edge_index


# --------------------------------------------------------------------
# Sampler paralel (Cython) - wrapper Python
# --------------------------------------------------------------------


class rw_sampling(GraphSampler):
    """
    Unbiased Random Walk sampler:
      - pilih size_root akar
      - jalan random sepanjang size_depth
      - ambil node yang tersentuh -> subgraf
    """

    def __init__(self, adj_train, node_train, size_subgraph, size_root, size_depth):
        self.size_root = int(size_root)
        self.size_depth = int(size_depth)
        size_subgraph = self.size_root * self.size_depth
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.cy_sampler = cy.RW(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.size_root,
            self.size_depth,
        )

    def preproc(self, **kwargs):
        pass


class edge_sampling(GraphSampler):
    """
    Edge sampler dengan probabilitas:
        p_{u,v} ∝ 1/deg(u) + 1/deg(v)
    """

    def __init__(self, adj_train, node_train, num_edges_subgraph):
        self.num_edges_subgraph = int(num_edges_subgraph)
        # estimasi #node per subgraf (kasar)
        self.size_subgraph = self.num_edges_subgraph * 2

        # derajat aman (float32) dan 1/deg aman (deg=0 -> 0)
        self.deg_train = np.array(adj_train.sum(1)).ravel().astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_deg = np.reciprocal(self.deg_train)
        inv_deg[~np.isfinite(inv_deg)] = 0.0

        # A' = D^{-1} A (row-normalized utk p_e)
        self.adj_train_norm = (
            scipy.sparse.dia_matrix((inv_deg, 0), shape=adj_train.shape)
            .dot(adj_train)
            .tocsr()
        )
        if self.adj_train_norm.dtype != np.float32:
            self.adj_train_norm = self.adj_train_norm.astype(np.float32)

        super().__init__(adj_train, node_train, self.size_subgraph, {})
        # cy sampler di-set setelah preproc (karena butuh edge_prob_tri)

        # set di sini setelah preproc
        self.cy_sampler = cy.Edge2(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.edge_prob_tri.row,
            self.edge_prob_tri.col,
            self.edge_prob_tri.data.cumsum(),
            self.num_edges_subgraph,
        )

    def preproc(self, **kwargs):
        """
        Hitung distribusi probabilitas edge simetris:
            P_e ∝ a_{u,v} + a_{v,u}
        lalu keep upper-triangular bagian saja.
        """
        nnz = self.adj_train.nnz
        self.edge_prob = scipy.sparse.csr_matrix(
            (
                np.zeros(nnz, dtype=np.float32),
                self.adj_train.indices,
                self.adj_train.indptr,
            ),
            shape=self.adj_train.shape,
        )
        # mulai dari A' (row-norm)
        self.edge_prob.data[:] = self.adj_train_norm.data[:]

        # tambahkan transpos (kolom-norm *implisit*), hasilnya simetris
        _adj_trans = scipy.sparse.csr_matrix.tocsc(self.adj_train_norm)
        self.edge_prob.data += _adj_trans.data  # P_e ∝ a_{u,v} + a_{v,u}

        # bersihkan nilai tak hingga (harusnya tidak ada)
        _mask = np.isfinite(self.edge_prob.data)
        if not np.all(_mask):
            self.edge_prob.data[~_mask] = 0.0

        # scaling sehingga sum(data) ≈ 2 * num_edges_subgraph
        total = self.edge_prob.data.sum()
        if total > 0:
            self.edge_prob.data *= 2.0 * self.num_edges_subgraph / total

        # simpan hanya segitiga atas (undirected)
        self.edge_prob_tri = (
            scipy.sparse.triu(self.edge_prob).astype(np.float32).tocoo()
        )


class mrw_sampling(GraphSampler):
    """
    Multi-dimensional Random Walk (MRW).
    """

    def __init__(
        self, adj_train, node_train, size_subgraph, size_frontier, max_deg=10000
    ):
        self.size_frontier = int(size_frontier)
        self.max_deg = int(max_deg)
        self.p_dist = None  # diisi saat preproc
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.name_sampler = "MRW"
        self.cy_sampler = cy.MRW(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.p_dist,
            self.max_deg,
            self.size_frontier,
            self.size_subgraph,
        )

    def preproc(self, **kwargs):
        # p_dist: bobot proporsional ke derajat (jumlah nilai baris)
        _adj = self.adj_train.tocsr()
        row_sums = np.asarray(_adj.sum(axis=1)).ravel()
        # casting ke int32/cumsum untuk Cython (batasi jika terlalu besar)
        p = row_sums.astype(np.int64)
        if p.sum() == 0:
            # fallback: semua sama
            p = np.ones_like(p, dtype=np.int64)
        # batasi supaya tidak overflow 32-bit saat Cython (heuristik)
        if p.sum() > 2**31 - 1:
            p = (p / p.sum() * (2**31 - 2)).astype(np.int64)
            p[p == 0] = 1
        self.p_dist = p.astype(np.int32)


class node_sampling(GraphSampler):
    """
    Node sampler (FastGCN-style).
    """

    def __init__(self, adj_train, node_train, size_subgraph):
        self.p_dist = None
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.cy_sampler = cy.Node(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.p_dist,
            self.size_subgraph,
        )

    def preproc(self, **kwargs):
        # Distribusi node ∝ sum(data) pada baris node_train
        _p = np.array(
            [
                self.adj_train.data[
                    self.adj_train.indptr[v] : self.adj_train.indptr[v + 1]
                ].sum()
                for v in self.node_train
            ],
            dtype=np.int64,
        )
        # cumsum untuk sampling cepat
        _p = _p.cumsum()
        if _p.size == 0 or _p[-1] == 0:
            # fallback kalau semua nol
            _p = np.arange(1, max(2, self.node_train.size + 1), dtype=np.int64)
        # batasi ke 32-bit
        if _p[-1] > 2**31 - 1:
            _p = (_p / _p[-1] * (2**31 - 1)).astype(np.int64)
            # pastikan strictly increasing
            _p = np.maximum.accumulate(np.maximum(_p, 1))
        self.p_dist = _p.astype(np.int32)


class full_batch_sampling(GraphSampler):
    """
    Bukan sampler, tapi kembalikan graf penuh (baseline).
    """

    def __init__(self, adj_train, node_train, size_subgraph):
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.cy_sampler = cy.FullBatch(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
        )


# --------------------------------------------
# Contoh sampler pure Python
# --------------------------------------------


class NodeSamplingVanillaPython(GraphSampler):
    """
    Contoh sampler Python murni: pilih node uniform lalu ambil subgraf node-induced.
    """

    def __init__(self, adj_train, node_train, size_subgraph):
        super().__init__(adj_train, node_train, size_subgraph, {})

    def par_sample(self, stage, **kwargs):
        node_ids = np.random.choice(self.node_train, self.size_subgraph)
        ret = self._helper_extract_subgraph(node_ids)
        ret = list(ret)
        # bungkus seperti output Cython: list of lists
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret

    def preproc(self, **kwargs):
        pass
