
import torch as th
from torch import nn
import dgl.function as fn

from  linear import TypedLinear


class qqq(nn.Module):


    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        regularizer=None,
        num_bases=None,
        bias=True,
        activation=None,
        self_loop=True,
        dropout=0.0,
        layer_norm=False,
    ):
        super().__init__()
        if regularizer is not None and num_bases is None:
            num_bases = num_rels
        self.linear_r = TypedLinear(
            in_feat, out_feat, num_rels, regularizer, num_bases
        )
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # TODO(minjie): consider remove those options in the future to make
        #   the module only about graph convolution.
        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(
                out_feat, elementwise_affine=True
            )

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout)

    def message(self, edges):
        """Message function."""
        m = self.linear_r(edges.src["h"], edges.data["etype"], self.presorted)
        if "norm" in edges.data:
            m = m * edges.data["norm"]
        return {"m": m}

    def forward(self, g, feat, etypes, norm=None, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor or list[int]
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        norm : torch.Tensor, optional
            An 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether the edges of the input graph have been sorted by their types.
            Forward on pre-sorted graph may be faster. Graphs created
            by :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for sorting edges manually.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        self.presorted = presorted
        with g.local_scope():
            g.srcdata["h"] = feat
            if norm is not None:
                g.edata["norm"] = norm
            g.edata["etype"] = etypes
            # message passing
            g.update_all(self.message, fn.sum("m", "h"))
            # apply bias and activation
            h = g.dstdata["h"]
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[: g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h
