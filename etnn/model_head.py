import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax

from etnn.layers import ETNNLayer
from etnn import utils, invariants


class ETNN(nn.Module):
    """
    The E(n)-Equivariant Topological Neural Network (ETNN) model.
    """

    def __init__(
        self,
        num_features_per_rank: dict[int, int],
        num_hidden: int,
        num_out: int,
        num_layers: int,
        adjacencies: list[str],
        initial_features: str,
        visible_dims: list[int] | None,
        normalize_invariants: bool,
        hausdorff_dists: bool = True,
        batch_norm: bool = False,
        dropout: float = 0.0,
        lean: bool = True,
        global_pool: bool = False,  # whether or not to use global pooling
        sparse_invariant_computation: bool = False,
        sparse_agg_max_cells: int = 100,  # maximum size to consider for diameter and hausdorff dists
        pos_update: bool = False,  # performs the equivariant position update, optional
    ) -> None:
        super().__init__()

        self.initial_features = initial_features

        # make inv_fts_map for backward compatibility
        self.num_invariants = 5 if hausdorff_dists else 3
        self.num_inv_fts_map = {k: self.num_invariants for k in adjacencies}
        self.adjacencies = adjacencies
        self.normalize_invariants = normalize_invariants
        self.batch_norm = batch_norm
        self.lean = lean
        max_dim = max(num_features_per_rank.keys())
        self.global_pool = global_pool
        self.visible_dims = visible_dims
        self.pos_update = pos_update
        self.dropout = dropout

        # params for invariant computation
        self.sparse_invariant_computation = sparse_invariant_computation
        self.sparse_agg_max_cells = sparse_agg_max_cells
        self.hausdorff = hausdorff_dists
        self.cell_list_fmt = "list" if sparse_invariant_computation else "padded"

        if sparse_invariant_computation:
            self.inv_fun = invariants.compute_invariants_sparse
        else:
            self.inv_fun = invariants.compute_invariants

        # keep only adjacencies that are compatible with visible_dims
        if visible_dims is not None:
            self.adjacencies = []
            for adj in adjacencies:
                max_rank = max(int(rank) for rank in adj.split("_")[:2])
                if max_rank in visible_dims:
                    self.adjacencies.append(adj)
        else:
            self.visible_dims = list(range(max_dim + 1))
            self.adjacencies = adjacencies

        # layers
        if self.normalize_invariants:
            self.inv_normalizer = nn.ModuleDict(
                {
                    adj: nn.BatchNorm1d(self.num_inv_fts_map[adj], affine=False)
                    for adj in self.adjacencies
                }
            )

        embedders = {}
        for dim in self.visible_dims:
            embedder_layers = [nn.Linear(num_features_per_rank[dim], num_hidden)]
            if self.batch_norm:
                embedder_layers.append(nn.BatchNorm1d(num_hidden))
            embedders[str(dim)] = nn.Sequential(*embedder_layers)
        self.feature_embedding = nn.ModuleDict(embedders)

        self.layers = nn.ModuleList(
            [
                ETNNLayer(
                    self.adjacencies,
                    self.visible_dims,
                    num_hidden,
                    self.num_inv_fts_map,
                    self.batch_norm,
                    self.lean,
                    self.pos_update,
                )
                for _ in range(num_layers)
            ]
        )

        self.pre_pool = nn.ModuleDict()

        for dim in self.visible_dims:
            if self.global_pool:
                if not self.lean:
                    self.pre_pool[str(dim)] = nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                        nn.Linear(num_hidden, num_hidden),
                    )
                else:
                    self.pre_pool[str(dim)] = nn.Linear(num_hidden, num_hidden)
            else:
                if not self.lean:
                    self.pre_pool[str(dim)] = nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                        nn.Linear(num_hidden, num_out),
                    )
                else:
                    self.pre_pool[str(dim)] = nn.Linear(num_hidden, num_out)

        if self.global_pool:
            #self.head = GatedMultiAggHead(
            #    num_hidden=num_hidden,
            #    visible_dims=self.visible_dims,
            #    num_out=num_out,
            #    dropout=self.dropout if self.dropout > 0 else 0.1,  # small default
            #    lean=self.lean,
            #)
            self.head = AttentiveHead(
                num_hidden=num_hidden,
                visible_dims=self.visible_dims,
                num_out=num_out,
                dropout=self.dropout if self.dropout > 0 else 0.1,
                use_multiagg=True,   
                attn_tau=1.0,                 
                use_attn_layernorm=False,
            )

    def forward(self, graph: Data) -> Tensor:
        device = graph.pos.device

        cell_ind = {
            str(i): graph.cell_list(i, format=self.cell_list_fmt)
            for i in self.visible_dims
        }

        adj = {
            adj_type: getattr(graph, f"adj_{adj_type}")
            for adj_type in self.adjacencies
            if hasattr(graph, f"adj_{adj_type}")
        }

        # compute initial features
        features = {}
        for feature_type in self.initial_features:
            features[feature_type] = {}
            for i in self.visible_dims:
                if feature_type == "node":
                    features[feature_type][str(i)] = invariants.compute_centroids(
                        cell_ind[str(i)], graph.x
                    )
                elif feature_type == "mem":
                    mem = {i: getattr(graph, f"mem_{i}") for i in self.visible_dims}
                    features[feature_type][str(i)] = mem[i].float()
                elif feature_type == "hetero":
                    features[feature_type][str(i)] = getattr(graph, f"x_{i}")

        x = {
            str(i): torch.cat(
                [
                    features[feature_type][str(i)]
                    for feature_type in self.initial_features
                ],
                dim=1,
            )
            for i in self.visible_dims
        }

        # if using sparse invariant computation, obtain indces
        inv_comp_kwargs = {
            "cell_ind": cell_ind,
            "adj": adj,
            "hausdorff": self.hausdorff,
        }
        if self.sparse_invariant_computation:
            agg_indices, _ = invariants.sparse_computation_indices_from_cc(
                cell_ind, adj, self.sparse_agg_max_cells
            )
            inv_comp_kwargs["rank_agg_indices"] = agg_indices

        # embed features and E(n) invariant information
        pos = graph.pos
        x = {dim: self.feature_embedding[dim](feature) for dim, feature in x.items()}
        inv = self.inv_fun(pos, **inv_comp_kwargs)

        if self.normalize_invariants:
            inv = {
                adj: self.inv_normalizer[adj](feature) for adj, feature in inv.items()
            }

        # message passing
        for layer in self.layers:
            x, pos = layer(x, adj, inv, pos)
            if self.pos_update:
                inv = self.inv_fun(pos, **inv_comp_kwargs)
                if self.normalize_invariants:
                    inv = {
                        adj: self.inv_normalizer[adj](feature)
                        for adj, feature in inv.items()
                    }
            # apply dropout if needed (only during training)
            if self.dropout > 0 and self.training:
                x = {
                    dim: nn.functional.dropout(feature, p=self.dropout)
                    for dim, feature in x.items()
                }

        # read out
        out = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}

        if self.global_pool:
            # create one dummy node with all features equal to zero for each graph and each rank
            cell_batch = {
                str(i): utils.slices_to_pointer(graph._slice_dict[f"slices_{i}"]).to(device)
                for i in self.visible_dims
            }
            out = self.head(out, cell_batch)   # -> [B, num_out]
            #state = torch.cat(
            #    tuple([feature for dim, feature in out.items()]),
            #    dim=1,
            #)
            #out = self.post_pool(state)
            #out = torch.squeeze(out, -1)

        return out

    def __str__(self):
        return f"ETNN ({self.type})"
    
    @torch.no_grad()
    def predict_with_maps(self, graph: Data):
        device = graph.pos.device
        cell_ind = {str(i): graph.cell_list(i, format=self.cell_list_fmt) for i in self.visible_dims}
        adj = {a: getattr(graph, f"adj_{a}") for a in self.adjacencies if hasattr(graph, f"adj_{a}")}

        # (same feature/invariant pipeline as forward)
        features = {}
        for feature_type in self.initial_features:
            features[feature_type] = {}
            for i in self.visible_dims:
                if feature_type == "node":
                    features[feature_type][str(i)] = invariants.compute_centroids(cell_ind[str(i)], graph.x)
                elif feature_type == "mem":
                    mem = {i: getattr(graph, f"mem_{i}") for i in self.visible_dims}
                    features[feature_type][str(i)] = mem[i].float()
                elif feature_type == "hetero":
                    features[feature_type][str(i)] = getattr(graph, f"x_{i}")

        x = {str(i): torch.cat([features[ft][str(i)] for ft in self.initial_features], dim=1)
             for i in self.visible_dims}

        pos = graph.pos
        x = {dim: self.feature_embedding[dim](feat) for dim, feat in x.items()}
        inv_comp_kwargs = {"cell_ind": cell_ind, "adj": adj, "hausdorff": self.hausdorff}
        if self.sparse_invariant_computation:
            agg_indices, _ = invariants.sparse_computation_indices_from_cc(cell_ind, adj, self.sparse_agg_max_cells)
            inv_comp_kwargs["rank_agg_indices"] = agg_indices
        inv = self.inv_fun(pos, **inv_comp_kwargs)
        if self.normalize_invariants:
            inv = {a: self.inv_normalizer[a](f) for a, f in inv.items()}

        for layer in self.layers:
            x, pos = layer(x, adj, inv, pos)
            if self.pos_update:
                inv = self.inv_fun(pos, **inv_comp_kwargs)
                if self.normalize_invariants:
                    inv = {a: self.inv_normalizer[a](f) for a, f in inv.items()}
            if self.training and self.dropout > 0:
                x = {d: nn.functional.dropout(feat, p=self.dropout) for d, feat in x.items()}

        out = {d: self.pre_pool[d](feat) for d, feat in x.items()}
        cell_batch = {str(i): utils.slices_to_pointer(graph._slice_dict[f"slices_{i}"]).to(device)
                      for i in self.visible_dims}
        return self.head(out, cell_batch, return_maps=True)  # -> (pred, maps)
    
    def _init_linear_or_norm(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    
    def reinit_head(self):
        """
        For fine-tuning purposes.
        Re-initialize ONLY the head.
        """
        if self.global_pool:
            # reinit everything inside the head
            self.head.apply(self._init_linear_or_norm)
        else:
            raise NotImplementedError
    
    def get_head_modules(self):
        return [self.head]
    
    def get_backbone_modules(self):
        mods = [self.feature_embedding, self.layers]
        # invariants normalizer is not part of 'head'
        if getattr(self, "inv_normalizer", None) is not None:
            mods.append(self.inv_normalizer)
        # pre_pool is backbone when global_pool=True (projects to H); head consumes it
        if self.global_pool:
            mods.append(self.pre_pool)
        return mods
    
    def get_head_parameters(self):
        for m in self.get_head_modules():
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    def get_backbone_parameters(self):
        for m in self.get_backbone_modules():
            for p in m.parameters():
                if p.requires_grad:
                    yield p
    
    def freeze_backbone(self):
        for p in self.get_backbone_parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.get_backbone_parameters():
            p.requires_grad = True

    def set_backbone_eval(self, flag: bool):
        """Put backbone in eval() during freeze so BatchNorm running stats dont update."""
        mods = [self.feature_embedding, self.layers, self.pre_pool]
        if getattr(self, "inv_normalizer", None) is not None:
            mods.append(self.inv_normalizer)
        for m in mods:
            m.eval() if flag else m.train()



class GatedMultiAggHead(nn.Module):
    """
    Per-rank gated multi-aggregation readout:
      For each visible rank dim:
        h -> pre-proj (already done in ETNN.pre_pool)
        gate = sigmoid(W_g h)
        pools = [sum(h), mean(h), max(h), sum(gate*h)]
        rank_vec = Linear(concat(pools)) -> R^H
      Concat rank_vec across ranks -> R * H
      Final MLP -> R^{num_out}
    """
    # TODO: WIDEN HEAD AND THEN NARROW AGAIN
    
    def __init__(self, num_hidden: int, visible_dims: list[int], num_out: int, dropout: float = 0.1, lean: bool = True):
        super().__init__()
        self.visible_dims = [str(d) for d in visible_dims]
        self.num_hidden = num_hidden
        self.dropout = dropout

        # one small gate and projection per rank
        self.gates = nn.ModuleDict({
            d: nn.Linear(num_hidden, 1) for d in self.visible_dims
        })
        # concat of [sum, mean, max, gated-sum] -> 4 * H -> H
        self.rank_proj = nn.ModuleDict({
            d: nn.Linear(4 * num_hidden, num_hidden) for d in self.visible_dims
        })

        # final head MLP
        layers = [
            nn.LayerNorm(len(self.visible_dims) * num_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(len(self.visible_dims) * num_hidden, num_hidden),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(num_hidden, num_out))
        self.final = nn.Sequential(*layers)

    def forward(self, per_rank_feats: dict[str, torch.Tensor], per_rank_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # per_rank_feats[dim]: [N_dim, H] after ETNN.pre_pool
        # per_rank_batch[dim]: [N_dim] mapping nodes/cells to graph index
        rank_vecs = []
        for d in sorted(self.visible_dims):  # Ensure deterministic iteration order
            h = per_rank_feats[d]                     # [N, H]
            b = per_rank_batch[d]                     # [N]
            g = torch.sigmoid(self.gates[d](h))       # [N, 1]

            sum_pool   = global_add_pool(h, b)        # [B, H]
            mean_pool  = global_mean_pool(h, b)       # [B, H]
            max_pool   = global_max_pool(h, b)        # [B, H]
            gsum_pool  = global_add_pool(g * h, b)    # [B, H]

            agg = torch.cat([sum_pool, mean_pool, max_pool, gsum_pool], dim=1)  # [B, 4H]
            rank_vec = self.rank_proj[d](agg)                                   # [B, H]
            rank_vecs.append(rank_vec)

        state = torch.cat(rank_vecs, dim=1)           # [B, R*H]
        out = self.final(state)                       # [B, num_out]
        return out.squeeze(-1) if out.shape[-1] == 1 else out
    

class GlobalAdditiveAttention(nn.Module):
    def __init__(self, hidden: int, tau: float = 1.0, use_layernorm: bool = False):
        super().__init__()
        self.tau = tau
        self.pre_score_norm = nn.LayerNorm(hidden) if use_layernorm else None
        self.score = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor, batch: torch.Tensor, return_weights: bool = False):
        # h: [N, H], batch: [N] graph ids
        if self.pre_score_norm is not None:
            h = self.pre_score_norm(h)
        s = self.score(h).squeeze(-1) / self.tau    # [N]
        
        # Use deterministic softmax instead of torch_geometric.utils.softmax
        a = torch.zeros_like(s)
        unique_graphs = torch.unique(batch, sorted=True)  # Use sorted=True for determinism
        for graph_id in unique_graphs:
            mask = (batch == graph_id)
            if mask.sum() > 0:
                a[mask] = torch.softmax(s[mask], dim=0)
        
        pooled = global_add_pool(a.unsqueeze(-1) * h, batch)
        return (pooled, a) if return_weights else pooled


class AttentiveHead(nn.Module):
    """
    Per-rank attention readout:
      For each visible rank d:
        att = Σ softmax(score(h_i)) * h_i      (additive attention)
        (optionally) concat with sum/mean/max
        rank_vec = Linear(concat) -> R^H
      Concat rank_vec across ranks -> R*H
      Final MLP -> R^{num_out}
    """
    def __init__(self, num_hidden: int, visible_dims: list[int], num_out: int,
                 dropout: float = 0.1, use_multiagg: bool = True,
                 attn_tau: float = 1.0, use_attn_layernorm: bool = False):
        super().__init__()
        self.visible_dims = [str(d) for d in visible_dims]
        self.num_hidden = num_hidden
        self.use_multiagg = use_multiagg
        self.dropout = dropout  # Store dropout value for manual control

        # per-rank attention scorer
        self.attn = nn.ModuleDict({
            d: GlobalAdditiveAttention(num_hidden, tau=attn_tau, use_layernorm=use_attn_layernorm) 
            for d in self.visible_dims
        })

        in_per_rank = num_hidden * (4 if use_multiagg else 1)  # [sum, mean, max, att] or just [att]
        self.rank_proj = nn.ModuleDict({d: nn.Linear(in_per_rank, num_hidden) for d in self.visible_dims})

        layers = [
            nn.LayerNorm(len(self.visible_dims) * num_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(len(self.visible_dims) * num_hidden, num_hidden),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(num_hidden, num_out))
        self.final = nn.Sequential(*layers)

    def forward(self, per_rank_feats: dict[str, torch.Tensor],
                per_rank_batch: dict[str, torch.Tensor],
                return_maps: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
        rank_vecs = []
        maps = {} if return_maps else None
        for d in self.visible_dims:
            h = per_rank_feats[d]       # [N, H]
            b = per_rank_batch[d]       # [N]
            if return_maps:
                att_pool, a = self.attn[d](h, b, return_weights=True)  # a: [N]
                maps[d] = {"weights": a, "batch": b}
            else:
                att_pool = self.attn[d](h, b) # [B, H]
            if self.use_multiagg:
                sum_pool  = global_add_pool(h, b)
                mean_pool = global_mean_pool(h, b)
                max_pool  = global_max_pool(h, b)
                agg = torch.cat([sum_pool, mean_pool, max_pool, att_pool], dim=1)  # [B, 4H]
            else:
                agg = att_pool  # [B, H]

            rank_vecs.append(self.rank_proj[d](agg))  # [B, H]

        state = torch.cat(rank_vecs, dim=1)  # [B, R*H]
        
        # Apply final layers with proper dropout control
        if self.training and self.dropout > 0:
            # During training, use the sequential with dropout
            out = self.final(state)
        else:
            # During evaluation, manually apply layers without dropout
            x = state
            for layer in self.final:
                if isinstance(layer, nn.Dropout):
                    continue  # Skip dropout during evaluation
                x = layer(x)
            out = x
            
        return out.squeeze(-1) if out.shape[-1] == 1 else out
    


