#!/usr/bin/python3
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
import utils
from utils import get_param
from decoder import ConvE
from dgl.nn.pytorch import RelGraphConv


class SE_GNN(nn.Module):
    def __init__(self, n_ent, h_dim,out_dim, n_rel,):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']

        # entity embedding
        self.ent_emb = get_param(self.n_ent, h_dim)

        # gnn layer
        self.kg_n_layer = self.cfg.kg_layer
        # semantic layer
        self.semantic_layers = nn.ModuleList([SemanticLayer(n_ent, h_dim, out_dim, n_rel) for _ in range(self.kg_n_layer-1)])
        # structure layer
        self.structure_layers = nn.ModuleList([StructureLayer(n_ent, h_dim, out_dim, n_rel) for _ in range(self.kg_n_layer)])

        self.rel_emb = get_param(self.n_rel * 2, h_dim)
        self.rel_layers = nn.ModuleList([RelLayer(n_ent, h_dim, out_dim, n_rel) for _ in range(self.kg_n_layer)])

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)
        # loss
        self.bce = nn.BCELoss()

        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)
        self.act = nn.Tanh()


    def forward(self, h_id, r_id, kg):
        """
        matching computation between query (h, r) and answer t.
        :param h_id: head entity id, (bs, )
        :param r_id: relation id, (bs, )
        :param kg: aggregation graph
        :return: matching score, (bs, n_ent)
        """
        # aggregate embedding
        ent_emb, rel_emb = self.aggragate_emb(kg)

        head = ent_emb[h_id]
        rel = rel_emb[r_id]
        # (bs, n_ent)
        score = self.predictor(head, rel, ent_emb)

        return score

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss

    # 更新 e = e + hr + he
    # 更新 hr = Wr
    def aggragate_emb(self, kg):
        """
        aggregate embedding.
        :param kg:
        :return:
        """
        ent_emb = self.ent_emb
        rel_emb = self.rel_emb

        for semantic_layer, structure_layer, rel_layer in zip(self.semantic_layers, self.structure_layers, self.rel_layers):
            ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)
            semantic_ent_emb = semantic_layer(kg, ent_emb, rel_emb)
            structure_ent_emb = structure_layer(kg, ent_emb, rel_emb)
            ent_emb = ent_emb + semantic_ent_emb + structure_ent_emb
            play_rel_emb = rel_layer(kg, ent_emb, rel_emb)
            rel_emb = rel_emb + play_rel_emb
            #rel_emb =  play_rel_emb
        return ent_emb, rel_emb

class RelLayer(nn.Module):
    def __init__(self, n_ent, h_dim, out_dim, n_rel, num_layers = 1):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.r_dim = h_dim
        self.batch = 3
        self.hidden = h_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(h_dim, out_dim, num_layers=self.num_layers, batch_first=True)
        self.act = nn.Tanh()

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]  # 节点数量
        assert rel_emb.shape[0] == 2 * self.n_rel  # 237 * 2

        with kg.local_scope():
            #pred_rel_emb = rel_emb
            num_batches = self.n_rel * 2 // self.batch
            rels = rel_emb[:num_batches * self.batch, :]
            rels = rels.reshape((self.batch, num_batches, self.hidden)).transpose(0, 1)

            h = (get_param(self.num_layers, num_batches, self.hidden)).to(self.device)
            output, h = self.gru(rels, h)
            rel_emb = output.transpose(0, 1).reshape((num_batches * self.batch, self.hidden))
            pred_rel_emb = rel_emb
            pred_rel_emb = self.act(pred_rel_emb)
        return pred_rel_emb


class StructureLayer(nn.Module):
    def __init__(self, n_ent, h_dim, out_dim, n_rel):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.comp_op = self.cfg.comp_op
        assert self.comp_op in ['add', 'mul']

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel
        rel_id = kg.edata['rel_id']
        kg.edata['emb'] = rel_emb[rel_id]
        with kg.local_scope():
            kg.ndata['emb'] = ent_emb

            if self.cfg.comp_op == 'add':
                # e+r
                kg.apply_edges(fn.u_add_e('emb', 'emb', 'comp_emb'))
            elif self.cfg.comp_op == 'mul':
                # e*r
                kg.apply_edges(fn.u_mul_e('emb', 'emb', 'comp_emb'))
            else:
                raise NotImplementedError

            # attention
            kg.apply_edges(fn.e_dot_v('comp_emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            # agg
            kg.edata['comp_emb'] = kg.edata['comp_emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb

class SemanticLayer(nn.Module):
    def __init__(self, n_ent, h_dim, out_dim, n_rel,
                 regularizer="basis", num_bases=-1, dropout=0.,
                 self_loop=False, num_layers=2):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.batch = 3
        self.hidden = h_dim
        self.num_layers = num_layers
        if num_bases == -1:
            num_bases = n_rel * 2
        self.conv1 = RelGraphConv(h_dim, self.hidden, n_rel, regularizer,
                                  num_bases, self_loop)
        self.conv2 = RelGraphConv(self.hidden, out_dim, n_rel, regularizer,
                                  num_bases, self_loop)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh()
        #self.rnn = nn.RNN(h_dim, out_dim, num_layers=self.num_layers, batch_first=True)
        self.gru = nn.GRU(h_dim, out_dim, num_layers=self.num_layers, batch_first=True)
        #self.lstm = nn.LSTM(h_dim, out_dim, num_layers=self.num_layers, batch_first=True)
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]

        with kg.local_scope():
            # kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]
            kg.edata['norm'] = dgl.norm_by_dst(kg).unsqueeze(-1)

            num_batches = self.n_ent // self.batch
            nodes = ent_emb[:num_batches * self.batch, :]
            nodes = nodes.reshape((self.batch , num_batches, self.hidden)).transpose(0, 1)
            h = (get_param(self.num_layers, num_batches, self.hidden)).to(self.device)
            #output, h = self.rnn(nodes, h)
            output, h = self.gru(nodes, h)
            ent_emb = output.transpose(0, 1).reshape((num_batches * self.batch, self.hidden))
            kg.ndata['emb'] = ent_emb

            # LSTM
            """num_batches = self.n_ent // self.batch
            nodes = ent_emb[:num_batches * self.batch, :]
            nodes = nodes.reshape((self.batch, num_batches, self.hidden)).transpose(0, 1)

            h = torch.zeros(self.num_layers , num_batches, self.hidden).to(self.device) 
            c = torch.zeros(self.num_layers, num_batches, self.hidden).to(self.device)
            output, (h, c) = self.lstm(nodes, (h, c))
            ent_emb = output.transpose(0, 1).reshape((num_batches * self.batch, self.hidden))
            kg.ndata['emb'] = ent_emb"""

            kg.ndata['emb'] = self.conv1(kg, kg.ndata['emb'], rel_id, kg.edata['norm'])
            kg.ndata['emb'] = self.dropout(kg.ndata['emb'])
            kg.ndata['emb'] = self.conv2(kg, kg.ndata['emb'], rel_id, kg.edata['norm'])
            kg.ndata['emb'] = self.dropout(F.relu(kg.ndata['emb']))
            # kg.ndata['emb'] = self.gru(kg.ndata['emb'])
            neigh_ent_emb = kg.ndata['emb']

        return neigh_ent_emb




