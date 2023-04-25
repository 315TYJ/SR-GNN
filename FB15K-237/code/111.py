#
import self as self
import torch
import dgl.function as fn
import dgl
import torch.nn.functional as F
from dgl import nn
from dgl.nn.pytorch import RelGraphConv
import dgl
import numpy as np
import torch as th

from utils import get_param
kg = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))


ent_emb = get_param(5, 2)
print(ent_emb)
rel_emb = get_param(5, 2)
print(rel_emb)
kg.edata['emb'] = rel_emb
print(kg.edata['emb'])
kg.apply_edges(fn.e_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
print(ent_emb)
print(rel_emb)
kg.ndata['emb'] = ent_emb
# 每个邻接关系rj对节点ei的重要程度α = softmax()
kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
# agg
# r*α
kg.edata['emb'] = kg.edata['emb'] * kg.edata['norm']
# fn.copy_e 将边特征(r*α)复制一遍，再生成一个属性'm'存放embedding
# fn.sum把节点收到的消息m进行sum聚合，再把结果赋值给边的'neigh'特征 ∑（r*α）
kg.update_all(fn.copy_e('emb', 'm'), fn.sum('m', 'neigh'))

# ∑（r*α）
neigh_ent_emb = kg.ndata['neigh']

# ∑（W*r*α）
neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)
print(neigh_ent_emb)



