import logging
import numpy as np
import torch
from torch import tensor
from os.path import join
import utils
from typing import Tuple, List
from torch.utils.data import Dataset
from itertools import chain
import dgl
from collections import defaultdict
import math


def construct_dict(dir_path):
    """
    construct the entity, relation dict
    :param dir_path: data directory path
    :return:
    """
    ent2id, rel2id = dict(), dict()

    # index entities / relations in the occurence order in train, valid and test set
    train_path, valid_path, test_path = join(dir_path, 'train.txt'), join(dir_path, 'valid.txt'), join(dir_path, 'test.txt')
    for path in [train_path, valid_path, test_path]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.split('\t')
                t = t[:-1]  # remove \n
                if h not in ent2id:
                    ent2id[h] = len(ent2id)
                if t not in ent2id:
                    ent2id[t] = len(ent2id)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)

    # arrange the items in id order
    ent2id, rel2id = dict(sorted(ent2id.items(), key=lambda x: x[1])), dict(sorted(rel2id.items(), key=lambda x: x[1]))
    
    return ent2id, rel2id


def read_data(set_flag):
    """
    read data from file
    :param set_flag: train / valid / test set flag
    :return:
    """
    assert set_flag in [
        'train', 'valid', 'test',
        ['train', 'valid'], ['train', 'valid', 'test']
    ]
    cfg = utils.get_global_config()
    dir_p = join(cfg.dataset_dir, cfg.dataset)
    ent2id, rel2id = construct_dict(dir_p)

    # read the file
    if set_flag in ['train', 'valid', 'test']:
        path = join(dir_p, '{}.txt'.format(set_flag))
        file = open(path, 'r', encoding='utf-8')
    elif set_flag == ['train', 'valid']:
        path1 = join(dir_p, 'train.txt')
        path2 = join(dir_p, 'valid.txt')
        file1 = open(path1, 'r', encoding='utf-8')
        file2 = open(path2, 'r', encoding='utf-8')
        file = chain(file1, file2)
    elif set_flag == ['train', 'valid', 'test']:
        path1 = join(dir_p, 'train.txt')
        path2 = join(dir_p, 'valid.txt')
        path3 = join(dir_p, 'test.txt')
        file1 = open(path1, 'r', encoding='utf-8')
        file2 = open(path2, 'r', encoding='utf-8')
        file3 = open(path3, 'r', encoding='utf-8')
        file = chain(file1, file2, file3)
    else:
        raise NotImplementedError

    src_list = []
    dst_list = []
    rel_list = []
    pos_tails = defaultdict(set)
    pos_heads = defaultdict(set)
    pos_rels = defaultdict(set)

    for i, line in enumerate(file):
        h, r, t = line.strip().split('\t')
        h, r, t = ent2id[h], rel2id[r], ent2id[t]
        src_list.append(h)
        dst_list.append(t)
        rel_list.append(r)

        # format data in query-answer form
        # (h, r, ?) -> t, (?, r, t) -> h
        pos_tails[(h, r)].add(t)
        pos_heads[(r, t)].add(h)
        pos_rels[(h, t)].add(r)  # edge relation
        pos_rels[(t, h)].add(r+len(rel2id))  # inverse relations

    output_dict = {
        'src_list': src_list,
        'dst_list': dst_list,
        'rel_list': rel_list,
        'pos_tails': pos_tails,
        'pos_heads': pos_heads,
        'pos_rels': pos_rels
    }

    return output_dict


def construct_kg(set_flag, directed=False):
    """
    construct kg.
    :param set_flag: train / valid / test的标志，使用哪一组数据来构建 KG.
    :param directed: whether add inverse version for each edge, to make a undirected graph.
    训练SE-GNN模型时为假，计算SE指标时为真。
    :return:
    """
    assert directed in [True, False]
    # {'dataset': 'WN18RR', 'epoch': 800, 'batch_size': 256, 'learning_rate': 0.0015, 'h_dim': 200, 'pred_rel_w': True, 'label_smooth': 0.1, 'kg_layer': 1, 'rm_rate': 0.5, 'ent_drop': 0.2, 'rel_drop': 0, 'comp_op': 'mul', 'bn': True, 'k_h': 10, 'k_w': 20, 'ent_drop_pred': 0.3, 'conv_drop': 0.1, 'fc_drop': 0.4, 'ker_sz': 7, 'out_channel': 250, 'dataset_list': ['FB15k_237', 'WN18RR'], 'device': 2, 'max_no_improve': 50, 'cpu_worker_num': 10, 'warmup_epoch': 5, 'project_dir': '/media/lab707/data/Tian/LaLaLa(dgl)', 'dataset_dir': '${project_dir}/data/dataset', 'output_dir': '${project_dir}/data/output/${dataset}'}
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    # DATASET_STATISTICS = dict(
    #     FB15k_237=dict(n_ent=14541, n_rel=237, n_train=272115, n_valid=17535, n_test=20466),
    #     WN18RR=dict(n_ent=40943, n_rel=11, n_train=86835, n_valid=3034, n_test=3134)
    # )
    n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']  # 边类型数 11


    d = read_data(set_flag)  # 'train'
    src_list, dst_list, rel_list = [], [], []


    # eid: record the edge id of queries, for randomly removing some edges when training
    # 记录 queries 的边ID，用于在训练时随机删除一些边
    eid = 0  #173670
    hr2eid, rt2eid = defaultdict(list), defaultdict(list)
    #
    for h, t, r in zip(d['src_list'], d['dst_list'], d['rel_list']):
        if directed:
            src_list.extend([h])
            dst_list.extend([t])
            rel_list.extend([r])
            hr2eid[(h, r)].extend([eid])
            rt2eid[(r, t)].extend([eid])
            eid += 1
        else:
            # include the inverse edges
            # inverse rel id: original id + rel num
            src_list.extend([h, t])
            dst_list.extend([t, h])
            rel_list.extend([r, r + n_rel])
            hr2eid[(h, r)].extend([eid, eid + 1])
            rt2eid[(r, t)].extend([eid, eid + 1])
            eid += 2

    src, dst, rel = tensor(src_list), tensor(dst_list), tensor(rel_list)
    srclist, dstlist, rellist = src_list, dst_list, rel_list
    triplets = torch.stack([src, rel, dst], dim=1)

    # src tensor：（173670，）tensor([    0,     1,     2,  ..., 29637, 22554,   774])
    # dst tensor：（173670，）tensor([    1,     0,     3,  ...,  3574,   774, 22554])
    # rel tensor: (173670, ) tensor([    0,    11,     1,  ...,   17,     5,    16])
    # hr2eid tensor: (62547, ) {(0,0):[0,1],(2,1):[2,3,25394,25395,90060,90061],...}  加自循环
    # rt2eid tensor: (40962, ) {(0,1):[0,1],(1,3):[2,3],(1,5):[4,5],...}
    return src, dst, rel, hr2eid, rt2eid


def get_kg(src, dst, rel, device):
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']  # n_ent : 40943
    # Graph(num_nodes=40943, num_edges=173670,
    #       ndata_schemes={}
    #       edata_schemes={'rel_id': Scheme(shape=(), dtype=torch.int64)})
    kg = dgl.graph((src, dst), num_nodes=n_ent)
    # {'rel_id': tensor([ 0, 11,  1,  ..., 17,  5, 16], device='cuda:2')}
    kg.edata['rel_id'] = rel
    kg.edata['etype'] = rel
    kg = kg.to(device)

    return kg


class TrainDataset(Dataset):
    """
    Training data is in query-answer format: (h, r) -> tails, (r, t) -> heads
    """
    def __init__(self, set_flag, hr2eid, rt2eid):
        assert set_flag in ['train', 'valid', 'test']
        logging.info('---Load Train Data---')
        self.cfg = utils.get_global_config()
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.d = read_data(set_flag)
        self.query = []
        self.label = []
        self.rm_edges = []
        self.set_scaling_weight = []

        # pred tails
        for k, v in self.d['pos_tails'].items():
            self.query.append((k[0], k[1], -1))
            self.label.append(list(v))
            # randomly removing edges later
            self.rm_edges.append(hr2eid[k])

        # pred heads
        for k, v in self.d['pos_heads'].items():
            # inverse relation
            self.query.append((k[1], k[0] + self.n_rel, -1))
            self.label.append(list(v))
            self.rm_edges.append(rt2eid[k])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        h, r, t = self.query[item]
        label = self.get_onehot_label(self.label[item])

        rm_edges = torch.tensor(self.rm_edges[item], dtype=torch.int64)
        rm_num = math.ceil(rm_edges.shape[0] * self.cfg.rm_rate)
        rm_inds = torch.randperm(rm_edges.shape[0])[:rm_num]
        rm_edges = rm_edges[rm_inds]

        return (h, r, t), label, rm_edges

    def get_onehot_label(self, label):
        onehot_label = torch.zeros(self.n_ent)
        onehot_label[label] = 1
        if self.cfg.label_smooth != 0.0:
            onehot_label = (1.0 - self.cfg.label_smooth) * onehot_label + (1.0 / self.n_ent)

        return onehot_label

    def get_pos_inds(self, label):
        pos_inds = torch.zeros(self.n_ent).to(torch.bool)
        pos_inds[label] = True
        return pos_inds

    @staticmethod
    def collate_fn(data):
        src = [d[0][0] for d in data]
        rel = [d[0][1] for d in data]
        dst = [d[0][2] for d in data]
        label = [d[1] for d in data]  # list of list
        rm_edges = [d[2] for d in data]

        src = torch.tensor(src, dtype=torch.int64)
        rel = torch.tensor(rel, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)  # (bs, )
        label = torch.stack(label, dim=0)  # (bs, n_ent)
        rm_edges = torch.cat(rm_edges, dim=0)  # (n_rm_edges, )

        return (src, rel, dst), label, rm_edges


class EvalDataset(Dataset):
    """
    Evaluating data is in triple format. Keep one for head-batch and tail-batch respectively,
    for computing each direction's metrics conveniently.
    """
    def __init__(self, set_flag, mode):
        assert set_flag in ['train', 'valid', 'test']
        assert mode in ['head_batch', 'tail_batch']
        self.cfg = utils.get_global_config()
        dataset = self.cfg.dataset
        self.mode = mode
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.d = read_data(set_flag)
        self.trip = [_ for _ in zip(self.d['src_list'], self.d['rel_list'], self.d['dst_list'])]
        self.d_all = read_data(['train', 'valid', 'test'])
        self.pos_t = self.d_all['pos_tails']
        self.pos_h = self.d_all['pos_heads']

    def __len__(self):
        return len(self.trip)

    def __getitem__(self, item):
        h, r, t = self.trip[item]

        if self.mode == 'tail_batch':
            # filter_bias, remove other ground truthes when ranking
            filter_bias = np.zeros(self.n_ent, dtype=np.float)
            filter_bias[list(self.pos_t[(h, r)])] = -float('inf')
            filter_bias[t] = 0.
        elif self.mode == 'head_batch':
            filter_bias = np.zeros(self.n_ent, dtype=np.float)
            filter_bias[list(self.pos_h[(r, t)])] = -float('inf')
            filter_bias[h] = 0.
            h, r, t = t, r+self.n_rel, h
        else:
            raise NotImplementedError

        return (h, r, t), filter_bias.tolist(), self.mode

    @staticmethod
    def collate_fn(data: List[Tuple[tuple, list, str]]):
        h = [d[0][0] for d in data]
        r = [d[0][1] for d in data]
        t = [d[0][2] for d in data]
        filter_bias = [d[1] for d in data]
        mode = data[0][-1]

        h = torch.tensor(h, dtype=torch.int64)
        r = torch.tensor(r, dtype=torch.int64)
        t = torch.tensor(t, dtype=torch.int64)
        filter_bias = torch.tensor(filter_bias, dtype=torch.float)

        return (h, r, t), filter_bias, mode


class BiDataloader(object):
    """
    Combine the head-batch and tail-batch evaluation dataloader.
    """
    def __init__(self, h_loader: iter, t_loader: iter):
        self.h_loader_len = len(h_loader)
        self.t_loader_len = len(t_loader)
        self.h_loader_step = 0
        self.t_loader_step = 0
        self.total_len = self.h_loader_len + self.t_loader_len
        self.h_loader = self.inf_loop(h_loader)
        self.t_loader = self.inf_loop(t_loader)
        self._step = 0

    def __next__(self):
        if self._step == self.total_len:
            # ensure that all the data of two dataloaders is accessed
            assert self.h_loader_step == self.h_loader_len
            assert self.t_loader_step == self.t_loader_len
            self._step = 0
            self.h_loader_step = 0
            self.t_loader_step = 0
            raise StopIteration
        if self._step % 2 == 0:
            # head-batch
            if self.h_loader_step < self.h_loader_len:
                data = next(self.h_loader)
                self.h_loader_step += 1
            else:
                # if head-batch complets, return tail-batch
                data = next(self.t_loader)
                self.t_loader_step += 1
        else:
            # tail-batch
            if self.t_loader_step < self.t_loader_len:
                data = next(self.t_loader)
                self.t_loader_step += 1
            else:
                data = next(self.h_loader)
                self.h_loader_step += 1
        self._step += 1
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return self.total_len

    @staticmethod
    def inf_loop(dataloader):
        """
        infinite loop
        :param dataloader:
        :return:
        """
        while True:
            for data in dataloader:
                yield data
