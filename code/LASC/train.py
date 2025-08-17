# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import argparse
import time
import os, sys
import os.path as osp
import shutil
import copy as cp
from tqdm import tqdm
from functools import partial
import psutil
import pdb
import numpy as np
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import scipy.sparse as ssp
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PygDataLoader
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from data.dgl_datasetnew import DglDataset
from data.pyg_dataset import pyg_dataset

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from torch_geometric.datasets import Planetoid
from preprocess import preprocess, preprocess_full
from graphormer.collator import collator
from utils import *
from models import *
from timer_guard import TimerGuard

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', 
                    stream=sys.stdout,
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

# ngnn code
import argparse
import datetime
import os
import sys
import time

import dgl
import torch
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset
from tqdm import tqdm
from ngnn_dataset import SEALOGBLDataset
import ngnn_models
import ngnn_utils
import wp_utils
import seal18_utils
import pandas as pd
import numpy as np

# wp type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2none(v):
    if v.lower()=='none':
        return None
    else:
        return str(v)



def train(num_datas):
    model.train()

    y_pred, y_true = torch.zeros([num_datas]), torch.zeros([num_datas])
    start = 0
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
        if args.ngnn_code:  # ngnn_code
            g, z, x, edge_weights, y = [
                item.to(device) if item is not None else None for item in data
            ]
            # import pdb; pdb.set_trace()
            # g.to(device)
            if 'Graphormer' in args.model:
                g.attn_bias = g.attn_bias.to(device)
                g.edge_index = g.edge_index.to(device)
                g.x = g.x.to(device)
                g.z = g.z.to(device)
                if args.use_len_spd:
                    g.len_shortest_path = g.len_shortest_path.to(device)
                if args.use_num_spd:
                    g.num_shortest_path = g.num_shortest_path.to(device)
                if args.use_cnb_jac:
                    g.undir_jac = g.undir_jac.to(device)
                if args.use_cnb_aa:
                    g.undir_aa = g.undir_aa.to(device)
                if args.use_cnb_ra:
                    g.undir_ra = g.undir_ra.to(device)
                if args.use_degree:
                    g.undir_degree = g.undir_degree.to(device)
                    if directed:
                        g.in_degree = g.in_degree.to(device)
                        g.out_degree = g.out_degree.to(device)

            num_datas_in_batch = y.numel()  # sieg
            optimizer.zero_grad()
            logits = model(g, z, x, edge_weight=edge_weights)
            loss = BCEWithLogitsLoss()(logits.view(-1), y.to(torch.float))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * g.batch_size
            # sieg
            end = min(start+num_datas_in_batch, num_datas)
            y_pred[start:end] = logits.view(-1).cpu().detach()
            y_true[start:end] = y.view(-1).cpu().to(torch.float)
            start = end
        else:  # sieg_code
            data = data.to(device)
            num_datas_in_batch = data.y.numel()
            optimizer.zero_grad()
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            new_data = data.clone()
            new_data.x = x
            new_data.edge_weight = edge_weight
            new_data.node_id = node_id
            logits = model(new_data)
            loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
            loss.backward()
            optimizer.step()
            if args.scheduler: scheduler.step()
            total_loss += loss.item() * data.num_graphs
            end = min(start+num_datas_in_batch, num_datas)
            y_pred[start:end] = logits.view(-1).cpu().detach()
            y_true[start:end] = data.y.view(-1).cpu().to(torch.float)
            start = end
    result = {}
    # result['Confuse'] = confusion_matrix(y_true, y_pred)
    # result['ACC'] = accuracy_score(y_true, y_pred)
    # result['Precision'] = precision_score(y_true, y_pred)
    # result['Recall'] = recall_score(y_true, y_pred)
    # result['F1'] = f1_score(y_true, y_pred)
 
    print(y_true, y_pred)
    result['AUC'] = roc_auc_score(y_true, y_pred)
    return total_loss / len(train_dataset), result


def test_model(model, loader, num_datas):
    model.eval()

    y_pred, y_true = torch.zeros([num_datas]), torch.zeros([num_datas])
    start = 0
    x_srcs, x_dsts = [], []
    for data in tqdm(loader, ncols=70):
        if args.ngnn_code:  # ngnn_code
            g, z, x, edge_weights, y = [
                item.to(device) if item is not None else None for item in data
            ]
            # import pdb; pdb.set_trace()
            # g.to(device)
            if 'Graphormer' in args.model:
                g.attn_bias = g.attn_bias.to(device)
                g.edge_index = g.edge_index.to(device)
                g.x = g.x.to(device)
                g.z = g.z.to(device)
                if args.use_len_spd:
                    g.len_shortest_path = g.len_shortest_path.to(device)
                if args.use_num_spd:
                    g.num_shortest_path = g.num_shortest_path.to(device)
                if args.use_cnb_jac:
                    g.undir_jac = g.undir_jac.to(device)
                if args.use_cnb_aa:
                    g.undir_aa = g.undir_aa.to(device)
                if args.use_cnb_ra:
                    g.undir_ra = g.undir_ra.to(device)
                if args.use_degree:
                    g.undir_degree = g.undir_degree.to(device)
                    if directed:
                        g.in_degree = g.in_degree.to(device)
                        g.out_degree = g.out_degree.to(device)

            num_datas_in_batch = y.numel()  # sieg
            logits = model(g, z, x, edge_weight=edge_weights)
            # sieg
            end = min(start+num_datas_in_batch, num_datas)
            y_pred[start:end] = logits.view(-1).cpu()
            y_true[start:end] = y.view(-1).cpu().to(torch.float)
            start = end
        else:  # sieg_code
            data = data.to(device)
            num_datas_in_batch = data.y.numel()
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            new_data = data.clone()
            new_data.x = x
            new_data.edge_weight = edge_weight
            new_data.node_id = node_id
            logits = model(new_data)
            end = min(start+num_datas_in_batch, num_datas)
            y_pred[start:end] = logits.view(-1).cpu()
            y_true[start:end] = data.y.view(-1).cpu().to(torch.float)
            start = end

        if args.output_logits and loader == final_test_loader:
            _, center_indices = np.unique(data.batch.cpu().numpy(), return_index=True)
            x_srcs += data.node_id[center_indices].tolist()
            x_dsts += data.node_id[center_indices+1].tolist()
    if args.output_logits and loader == final_test_loader:
        logits_file = log_file.replace('log.txt', 'logits.txt')
        with open(logits_file, 'a') as f:
            print(f'x_src: (len:{len(x_srcs)})', file=f)
            print(x_srcs, file=f)
            print(f'x_dst: (len:{len(x_dsts)})', file=f)
            print(x_dsts, file=f)
            print(f'y_pred: (len:{len(y_pred.tolist())})', file=f)
            print(y_pred.tolist(), file=f)
            print(f'y_true: (len:{len(y_true.tolist())})', file=f)
            print(y_true.tolist(), file=f)

    pos_test_pred = y_pred[y_true==1]
    neg_test_pred = y_pred[y_true==0]
    return y_pred, y_true, pos_test_pred, neg_test_pred


def eval_model(**kwargs):
    eval_metric = kwargs["eval_metric"]
    if eval_metric == 'hits':
        pos_val_pred = kwargs["pos_val_pred"]
        neg_val_pred = kwargs["neg_val_pred"]
        pos_test_pred = kwargs["pos_test_pred"]
        neg_test_pred = kwargs["neg_test_pred"]
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif eval_metric == 'mrr':
        pos_val_pred = kwargs["pos_val_pred"]
        neg_val_pred = kwargs["neg_val_pred"]
        pos_test_pred = kwargs["pos_test_pred"]
        neg_test_pred = kwargs["neg_test_pred"]
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif eval_metric == 'auc':
        val_pred = kwargs["val_pred"]
        val_true = kwargs["val_true"]
        test_pred = kwargs["test_pred"]
        test_true = kwargs["test_true"]
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results

@torch.no_grad()
def test(eval_metric):
    model.eval()

    val_pred, val_true, pos_val_pred, neg_val_pred = test_model(model, val_loader, len(val_dataset))

    test_pred, test_true, pos_test_pred, neg_test_pred = test_model(model, test_loader, len(test_dataset))

    result = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                      val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric=eval_metric)
    if eval_metric != 'auc':
        result_auc = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                          val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric='auc')
        for key in result_auc.keys():
            result[key] = result_auc[key]
    return result

@torch.no_grad()
def final_test(eval_metric):
    model.eval()

    val_pred, val_true, pos_val_pred, neg_val_pred = test_model(model, final_val_loader, len(final_val_dataset))

    test_pred, test_true, pos_test_pred, neg_test_pred = test_model(model, final_test_loader, len(final_test_dataset))

    result = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                      val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric=eval_metric)
    if eval_metric != 'auc':
        result_auc = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                          val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric='auc')
        for key in result_auc.keys():
            result[key] = result_auc[key]
    return result

@torch.no_grad()
def test_multiple_models_origin(models, eval_metric):
    num_models = len(models)
    for m in models:
        m.eval()

    y_preds, y_trues = [[] for _ in range(num_models)], [[] for _ in range(num_models)]
    for data in tqdm(val_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, model in enumerate(models):
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_preds[i].append(logits.view(-1).cpu())
            y_trues[i].append(data.y.view(-1).cpu().to(torch.float))
    val_preds = [torch.cat(y_preds[i]) for i in range(num_models)]
    val_trues = [torch.cat(y_trues[i]) for i in range(num_models)]
    pos_val_preds = [val_preds[i][val_trues[i]==1] for i in range(num_models)]
    neg_val_preds = [val_preds[i][val_trues[i]==0] for i in range(num_models)]
    mem = psutil.virtual_memory()
    print(f' after val - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')

    y_preds, y_trues = [[] for _ in range(num_models)], [[] for _ in range(num_models)]
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, model in enumerate(models):
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_preds[i].append(logits.view(-1).cpu())
            y_trues[i].append(data.y.view(-1).cpu().to(torch.float))
    test_preds = [torch.cat(y_preds[i]) for i in range(num_models)]
    test_trues = [torch.cat(y_trues[i]) for i in range(num_models)]
    pos_test_preds = [test_preds[i][test_trues[i]==1] for i in range(num_models)]
    neg_test_preds = [test_preds[i][test_trues[i]==0] for i in range(num_models)]

    mem = psutil.virtual_memory()
    print(f' after test - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
    results = eval_multiple_models(num_models,
                                pos_val_preds=pos_val_preds, neg_val_preds=neg_val_preds, pos_test_preds=pos_test_preds, neg_test_preds=neg_test_preds,
                                val_preds=val_preds, val_trues=val_trues, test_preds=test_preds, test_trues=test_trues, eval_metric=eval_metric)
    if eval_metric != 'auc':
        results_auc = eval_multiple_models(num_models,
                                    pos_val_preds=pos_val_preds, neg_val_preds=neg_val_preds, pos_test_preds=pos_test_preds, neg_test_preds=neg_test_preds,
                                    val_preds=val_preds, val_trues=val_trues, test_preds=test_preds, test_trues=test_trues, eval_metric='auc')
        for i in range(num_models):
            for key in results_auc[i].keys():
                results[i][key] = results_auc[i][key]

    return results


@torch.no_grad()
def test_multiple_models(models, loader, num_datas):
    num_models = len(models)
    for m in models:
        m.eval()

    y_preds, y_trues = [torch.zeros([num_datas]) for _ in range(num_models)], [torch.zeros([num_datas]) for _ in range(num_models)]
    start = 0
    for data in tqdm(loader, ncols=70):
        data = data.to(device)
        num_datas_in_batch = data.y.numel()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        new_data = data.clone()
        new_data.x = x
        new_data.edge_weight = edge_weight
        new_data.node_id = node_id
        end = min(start+num_datas_in_batch, num_datas)
        for i, model in enumerate(models):
            logits = model(new_data)
            y_preds[i][start:end] = logits.view(-1).cpu()
            y_trues[i][start:end] = data.y.view(-1).cpu().to(torch.float)
        start = end
    pos_test_preds = [y_preds[i][y_trues[i]==1] for i in range(num_models)]
    neg_test_preds = [y_preds[i][y_trues[i]==0] for i in range(num_models)]

    mem = psutil.virtual_memory()
    print(f'       max - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
    return y_preds, y_trues, pos_test_preds, neg_test_preds


def eval_multiple_models(num_models, **kwargs):
    eval_metric = kwargs["eval_metric"]
    Results = []
    for i in range(num_models):
        if eval_metric == 'hits':
            pos_val_preds = kwargs["pos_val_preds"]
            neg_val_preds = kwargs["neg_val_preds"]
            pos_test_preds = kwargs["pos_test_preds"]
            neg_test_preds = kwargs["neg_test_preds"]
            Results.append(evaluate_hits(pos_val_preds[i], neg_val_preds[i], pos_test_preds[i], neg_test_preds[i]))
        elif eval_metric == 'mrr':
            pos_val_preds = kwargs["pos_val_preds"]
            neg_val_preds = kwargs["neg_val_preds"]
            pos_test_preds = kwargs["pos_test_preds"]
            neg_test_preds = kwargs["neg_test_preds"]
            Results.append(evaluate_mrr(pos_val_preds[i], neg_val_preds[i], pos_test_preds[i], neg_test_preds[i]))
        elif eval_metric == 'auc':
            val_preds = kwargs["val_preds"]
            val_trues = kwargs["val_trues"]
            test_preds = kwargs["test_preds"]
            test_trues = kwargs["test_trues"]
            Results.append(evaluate_auc(val_preds[i], val_trues[i], test_preds[i], test_trues[i]))

    return Results


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in args.eval_hits_K:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)

    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)
    # results['Confuse'] = (confusion_matrix(val_true, val_pred), confusion_matrix(test_true, test_pred))
    # results['ACC'] = (accuracy_score(val_true, val_pred), accuracy_score(test_true, test_pred))
    # torch.set_printoptions(threshold=10000)
    # print(val_true)
    # print(val_pred)
    # print(test_true)
    # print(test_pred)
    # print(accuracy_score(val_true, val_pred), accuracy_score(test_true, test_pred))
    results['Precision'] = (precision_score(val_true, val_pred), precision_score(test_true, test_pred))
    # print(precision_score(val_true, val_pred), precision_score(test_true, test_pred))
    results['Recall'] = (recall_score(val_true, val_pred), recall_score(test_true, test_pred))
    # print(recall_score(val_true, val_pred), recall_score(test_true, test_pred))
    results['F1'] = (f1_score(val_true, val_pred), f1_score(test_true, test_pred))
    # print(f1_score(val_true, val_pred), f1_score(test_true, test_pred))
   
    # 将 Tensors 移到 CPU 并转换为 NumPy 数组
    val_true_np = val_true.detach().cpu().numpy() if val_true.is_cuda else val_true.detach().numpy()
    val_pred_np = val_pred.detach().cpu().numpy() if val_pred.is_cuda else val_pred.detach().numpy()
    test_true_np = test_true.detach().cpu().numpy() if test_true.is_cuda else test_true.detach().numpy()
    test_pred_np = test_pred.detach().cpu().numpy() if test_pred.is_cuda else test_pred.detach().numpy()

    # 打印数组的长度进行检查
    print(len(val_true_np), len(val_pred_np), len(test_true_np), len(test_pred_np))
    # 获取当前时间戳，作为文件名的一部分
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    # # 确保所有数组长度一致
    # if len(val_true_np) == len(val_pred_np) == len(test_true_np) == len(test_pred_np):
    #     # 创建 DataFrame
    #     df = pd.DataFrame({
    #         "val_true": val_true_np,
    #         "val_pred": val_pred_np,
    #         "test_true": test_true_np,
    #         "test_pred": test_pred_np
    #     })

    #     # 将 DataFrame 存储为 CSV 文件
    #     df.to_csv("classification_results.csv", index=False)
    # else:

    #     # 获取最长长度
    #     max_length = max(len(val_true_np), len(val_pred_np), len(test_true_np), len(test_pred_np))
        
    #     # 填充较短的数组
    #     val_true_np = np.pad(val_true_np, (0, max_length - len(val_true_np)), 'constant', constant_values=np.nan)
    #     val_pred_np = np.pad(val_pred_np, (0, max_length - len(val_pred_np)), 'constant', constant_values=np.nan)
    #     test_true_np = np.pad(test_true_np, (0, max_length - len(test_true_np)), 'constant', constant_values=np.nan)
    #     test_pred_np = np.pad(test_pred_np, (0, max_length - len(test_pred_np)), 'constant', constant_values=np.nan)
        
    #     # 创建 DataFrame
    #     df = pd.DataFrame({
    #         "val_true": val_true_np,
    #         "val_pred": val_pred_np,
    #         "test_true": test_true_np,
    #         "test_pred": test_pred_np
    #     })

    #     # 将 DataFrame 存储为 CSV 文件
    #     df.to_csv("classification_results.csv", index=False)
    # 获取最长长度
    max_length = max(len(val_true_np), len(val_pred_np), len(test_true_np), len(test_pred_np))
    
    # 填充较短的数组
    val_true_np = np.pad(val_true_np, (0, max_length - len(val_true_np)), 'constant', constant_values=np.nan)
    val_pred_np = np.pad(val_pred_np, (0, max_length - len(val_pred_np)), 'constant', constant_values=np.nan)
    test_true_np = np.pad(test_true_np, (0, max_length - len(test_true_np)), 'constant', constant_values=np.nan)
    test_pred_np = np.pad(test_pred_np, (0, max_length - len(test_pred_np)), 'constant', constant_values=np.nan)
    
    # 创建 DataFrame
    df = pd.DataFrame({
        "val_true": val_true_np,
        "val_pred": val_pred_np,
        "test_true": test_true_np,
        "test_pred": test_pred_np
    })

    # 将 DataFrame 存储为 CSV 文件，文件名包含时间戳
    df.to_csv(f"acc_rusult/classification_results_{timestamp}.csv", index=False)

    return results

# Data settings
parser = argparse.ArgumentParser(description='OGBL (SEAL)')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--cmd_time', type=str, default='ignore_time')
parser.add_argument('--root', type=str, default='dataset',
                    help="root of dataset")
parser.add_argument('--dataset', type=str, default='TC-mixer')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets, do a fast data split")
# GNN settings
# parser.add_argument('--model', type=str, default='NGNNDGCNNGraphormer')
parser.add_argument('--model', type=str, default='GCNGraphormer_noNeigFeat')
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--mlp_hidden_channels', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)

# Subgraph extraction settings
parser.add_argument('--sample_type', type=int, default=0)
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, nargs='+', default=None)
parser.add_argument('--node_label', type=str, default='drnl', 
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', action='store_true',
                    help="whether to use raw node features as GNN input",default=True)
parser.add_argument('--use_feature_GT', action='store_true',
                    help="whether to use raw node features as GNN input",default=True)
parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN",default=None)
parser.add_argument('--use_rpe', action='store_true', help="whether to use RPE as GNN input")
parser.add_argument('--replacement', action='store_true', help="whether to enable replacement sampleing in random walk")
parser.add_argument('--trackback', action='store_true', help="whether to enabale trackback path searching in random walk")
parser.add_argument('--num_walk', type=int, default=200, help='total number of random walks')
parser.add_argument('--num_step', type=int, default=4, help='total steps of random walk')
parser.add_argument('--rpe_hidden_dim', type=int, default=16, help='dimension of RPE embedding')
parser.add_argument('--gravity_type', type=int, default=0)
parser.add_argument('--readout_type', type=int, default=0)
# Training settings
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--train_percent', type=float, default=60)
parser.add_argument('--val_percent', type=float, default=20)
parser.add_argument('--test_percent', type=float, default=20)
parser.add_argument('--final_val_percent', type=float, default=100)
parser.add_argument('--final_test_percent', type=float, default=100)
parser.add_argument('--dynamic_train', action='store_true', 
                    help="dynamically extract enclosing subgraphs on the fly", default=True)
parser.add_argument('--dynamic_val', action='store_true', default=True)
parser.add_argument('--dynamic_test', action='store_true', default=True)
parser.add_argument('--slice_type', type=int, default=0,
                    help="type of saving sampled subgraph in disk")
parser.add_argument('--num_workers', type=int, default=8, 
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--train_node_embedding', action='store_true',
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--dont_z_emb_agg', action='store_true')
parser.add_argument('--pretrained_node_embedding', type=str, default=None, 
                    help="load pretrained node embeddings as additional node features")
# Testing settings
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix', type=str, default='', 
                    help="an appendix to the data directory")
parser.add_argument('--save_appendix', type=str, default='', 
                    help="an appendix to the save directory")
parser.add_argument('--keep_old', action='store_true', 
                    help="do not overwrite old files in the save directory")
parser.add_argument('--continue_from', type=int, nargs='*', default=None, 
                    help="from which run and epoch checkpoint to continue training")
parser.add_argument('--part_continue_from', type=int, nargs='*', default=None, 
                    help="from which run and epoch checkpoint to continue training")
parser.add_argument('--output_logits', action='store_true')
parser.add_argument('--only_test', action='store_true', 
                    help="only test without training")
parser.add_argument('--only_final_test', action='store_true', 
                    help="only final test without training")
parser.add_argument('--test_multiple_models', type=str, nargs='+', default=[], 
                    help="test multiple models together")
parser.add_argument('--use_heuristic', type=str, default=None,
                    help="test a link prediction heuristic (CN or AA)")
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--use_len_spd', action='store_true', default=True)
parser.add_argument('--use_num_spd', action='store_true', default=True)
parser.add_argument('--use_cnb_jac', action='store_true', default=True)
parser.add_argument('--use_cnb_aa', action='store_true', default=True)
parser.add_argument('--use_cnb_ra', action='store_true', default=False)
parser.add_argument('--use_degree', action='store_true', default=False)
parser.add_argument('--grpe_cross', action='store_true', default=True)
parser.add_argument('--use_ignn', action='store_true', default=False)
parser.add_argument('--mul_bias', action='store_true', default=False,
                    help="add bias to attention if true else multiple")
parser.add_argument('--max_z', type=int, default=1000)  # set a large max_z so that every z has embeddings to look up

# ngnn_args
parser.add_argument('--ngnn_code', action='store_true', default=True)
parser.add_argument('--use_full_graphormer', action='store_true', default=False)

parser.add_argument(
    "--ngnn_type",
    type=str,
    default="all",
    choices=["none", "input", "hidden", "output", "all"],
    help="You can set this value from 'none', 'input', 'hidden' or 'all' " \
            "to apply NGNN to different GNN layers.",
)
parser.add_argument(
    "--num_ngnn_layers", type=int, default=2, choices=[1, 2]
)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument(
    "--test_topk",
    type=int,
    default=1,
    help="select best k models for full validation/test each run.",
)
parser.add_argument(
    "--eval_hits_K",
    type=int,
    nargs="*",
    default=[10],
    help="hits@K for each eval step; " \
            "only available for datasets with hits@xx as the eval metric",
)

# wp_args
parser.add_argument('--wp_code', action='store_true', default=False)
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--val-ratio', type=float, default=0.05,
                    help='ratio of validation links. If using the splitted data from SEAL,\
                     it is the ratio on the observed links, othewise, it is the ratio on the whole links.')
parser.add_argument('--practical-neg-sample', type=bool, default = False,
                    help='only see the train positive edges when sampling negative')
parser.add_argument('--wp-seed', type=int, default=1)
parser.add_argument('--drnl', type=str2bool, default=False,
                    help='whether to use drnl labeling')
parser.add_argument('--data-split-num',type=str, default='10',
                    help='If use-splitted is true, choose one of splitted data')
parser.add_argument('--observe-val-and-injection', type=str2bool, default = True,
                    help='whether to contain the validation set in the observed graph and apply injection trick')
parser.add_argument('--init-attribute', type=str2none, default='ones',
                    help='initial attribute for graphs without node attributes\
                    , options: n2v, one_hot, spc, ones, zeros, None')
parser.add_argument('--init-representation', type=str2none, default= None,
                    help='options: gic, vgae, argva, None')
parser.add_argument('--use-splitted', type=str2bool, default=True,
                    help='use the pre-splitted train/test data,\
                     if False, then make a random division')
parser.add_argument('--embedding-dim', type=int, default= 32,
                    help='Dimension of the initial node representation, default: 32)')


# seal18_args
parser.add_argument('--seal18_code', action='store_true', default=False)
parser.add_argument('--train-name', type=str, default=None)
parser.add_argument('--test-name', type=str, default=None)
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')

args = parser.parse_args()


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
if args.seed is not None: seed_torch(args.seed)

if args.max_nodes_per_hop is not None and len(args.max_nodes_per_hop) == 1:
    args.max_nodes_per_hop = args.max_nodes_per_hop[0]
# if args.max_nodes_per_hop is not None:
#     args.max_nodes_per_hop = None if args.max_nodes_per_hop < 0 else args.max_nodes_per_hop
if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.data_appendix == '':
    args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
    if args.use_valedges_as_input:
        args.data_appendix += '_uvai'
if args.use_heuristic is not None:
    args.runs = 1

args.res_dir = os.path.join('results/{}{}'.format(args.dataset, args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
if not args.keep_old:
    # Backup files.
    backup_root_dir = os.path.join(args.res_dir, 'src')
    root_dir = os.path.dirname(sys.argv[0])
    root_dir = './' if root_dir == '' else root_dir
    for sub_dir in ['', 'surel_gacc']:
        full_dir = os.path.join(root_dir, sub_dir)
        files = [f for f in os.listdir(full_dir) if os.path.isfile(os.path.join(full_dir, f)) and os.path.splitext(f)[1] in ['.py", ".c", ".cpp']]
        backup_dir = os.path.join(backup_root_dir, sub_dir)
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        for f in files:
            shutil.copy(os.path.join(full_dir, f), backup_dir)
log_file = os.path.join(args.res_dir, 'log.txt')
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
with open(log_file, 'a') as f:
    f.write('\n' + cmd_input)

if args.dataset.startswith('ETH-mixer'):
    args.eval_metric = 'mrr'
    directed = True
else:  # assume other datasets are undirected
    args.eval_metric = 'mrr'
    directed = True


if args.dataset.startswith('ogbl'):
    evaluator = Evaluator(name=args.dataset)
if args.eval_metric == 'hits':
    loggers = {
        f"Hits@{k}": Logger(args.runs, args) for k in args.eval_hits_K
    }
elif args.eval_metric == 'mrr':
    loggers = {
        'MRR': Logger(args.runs, args),
    }
elif args.eval_metric == 'auc':
    loggers = {
        'AUC': Logger(args.runs, args),
    }
    
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:0')
device = 'cpu' if args.device == -1 or not torch.cuda.is_available() else f'cuda:{args.device}'
device = torch.device(device)

if args.use_heuristic:
    # Test link prediction heuristics.
    num_nodes = data.num_nodes
    if 'edge_weight' in data and args.use_edge_weight:
        edge_weight = data.edge_weight.view(-1)
    else:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), 
                       shape=(num_nodes, num_nodes))

    pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge, 
                                                   data.edge_index, 
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge, 
                                                     data.edge_index, 
                                                     data.num_nodes)
    if directed:
        cn_types = ['undirected', 'in', 'out', 's2o', 'o2s']
    else:
        cn_types = ['in']

    pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge, cn_types=cn_types)
    neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge, cn_types=cn_types)
    pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge, cn_types=cn_types)
    neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge, cn_types=cn_types)

    for idx_type in range(len(cn_types)):
        cn_type = cn_types[idx_type]
        if args.eval_metric == 'hits':
            results = evaluate_hits(pos_val_pred[idx_type], neg_val_pred[idx_type], pos_test_pred[idx_type], neg_test_pred[idx_type])
        elif args.eval_metric == 'mrr':
            results = evaluate_mrr(pos_val_pred[idx_type], neg_val_pred[idx_type], pos_test_pred[idx_type], neg_test_pred[idx_type])
        elif args.eval_metric == 'auc':
            val_pred = torch.cat([pos_val_pred[idx_type], neg_val_pred[idx_type]])
            val_true = torch.cat([torch.ones(pos_val_pred[idx_type].size(0), dtype=int), 
                                  torch.zeros(neg_val_pred[idx_type].size(0), dtype=int)])
            test_pred = torch.cat([pos_test_pred[idx_type], neg_test_pred[idx_type]])
            test_true = torch.cat([torch.ones(pos_test_pred[idx_type].size(0), dtype=int), 
                                  torch.zeros(neg_test_pred[idx_type].size(0), dtype=int)])
            results = evaluate_auc(val_pred, val_true, test_pred, test_true)

        for key, result in results.items():
            loggers[key].reset()
            loggers[key].add_result(0, result)
        for key in loggers.keys():
            print(cn_type)
            print(key)
            loggers[key].print_statistics()
            with open(log_file, 'a') as f:
                print(cn_type, file=f)
                print(key, file=f)
                loggers[key].print_statistics(f=f)
    # pdb.set_trace()
    exit()

preprocess_func = preprocess_full if args.use_full_graphormer else preprocess
preprocess_fn = partial(preprocess_func,
                        grpe_cross=args.grpe_cross,
                        use_len_spd=args.use_len_spd,
                        use_num_spd=args.use_num_spd,
                        use_cnb_jac=args.use_cnb_jac,
                        use_cnb_aa=args.use_cnb_aa,
                        use_cnb_ra=args.use_cnb_ra,
                        use_degree=args.use_degree,
                        gravity_type=args.gravity_type,
                )  if args.model.find('Graphormer') != -1 else None

if args.ngnn_code:
    dataset = DglDataset()
    # dataset = DglLinkPropPredDataset(name=args.dataset)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]

    # Re-format the data of TC-mixer.
    if args.dataset == "TC-mixer":
        for k in ["train", "valid", "test"]:
            src = split_edge[k]["source_node"]
            tgt = split_edge[k]["target_node"]
            split_edge[k]["edge"] = torch.stack([src, tgt], dim=1)  # [86596, 2]
            if k != "train":
                tgt_neg = split_edge[k]["target_node_neg"]
                split_edge[k]["edge_neg"] = torch.stack(
                    [src[:, None].repeat(1, tgt_neg.size(1)), tgt_neg], dim=-1
                )  # [Ns, Nt, 2] [86596, 1000, 2]

    if not args.use_edge_weight and "weight" in graph.edata:
        del graph.edata["weight"]
    if not args.use_feature and "feat" in graph.ndata:
        del graph.ndata["feat"]

    data_appendix = "_rph{}".format("".join(str(args.ratio_per_hop).split(".")))  # ngnn
    path = f"data/_seal{data_appendix}"
    if not (args.dynamic_train or args.dynamic_val or args.dynamic_test):
        args.num_workers = 0

    dataset_pyg=pyg_dataset()
    data_pyg = dataset_pyg[0]
    # print(data_pyg, preprocess_fn, path, split_edge,args.train_percent,args.val_percent,args.test_percent,directed,args.final_val_percent,args.final_test_percent)
    train_dataset, val_dataset, test_dataset, final_val_dataset, final_test_dataset = [
        SEALOGBLDataset(
            data_pyg,
            preprocess_fn,
            path,
            graph,
            split_edge,
            percent=percent,
            split=split,
            ratio_per_hop=args.ratio_per_hop,
            directed=directed,
            dynamic=dynamic,
        )
        for percent, split, dynamic in zip(
            [
                args.train_percent,
                args.val_percent,
                args.test_percent,
                args.final_val_percent,
                args.final_test_percent,
            ],
            ["train", "valid", "test", "valid", "test"],
            [
                args.dynamic_train,
                args.dynamic_val,
                args.dynamic_test,
                args.dynamic_val,
                args.dynamic_test,
            ],
        )
    ]
    # import pdb; pdb.set_trace()

    def ogbl_collate_fn(batch):
        gs, zs, xs, ws, ys, g_noaugs = zip(*batch)
        # import pdb; pdb.set_trace()
        batched_g = dgl.batch(gs)
        z = torch.cat(zs, dim=0)
        if xs[0] is not None:
            x = torch.cat(xs, dim=0)
        else:
            x = None
        if ws[0] is not None:
            edge_weights = torch.cat(ws, dim=0)
        else:
            edge_weights = None
        y = torch.tensor(ys)

        if 'Graphormer' in args.model:
            batched_g.attn_bias = torch.cat([g_noaug.pair_attn_bias for g_noaug in g_noaugs], dim=0)
            batched_g.edge_index = torch.cat([g_noaug.pair_edge_idx for g_noaug in g_noaugs], dim=0)
            batched_g.x = torch.cat([g_noaug.pair_x for g_noaug in g_noaugs], dim=0)
            batched_g.z = torch.cat([g_noaug.pair_z for g_noaug in g_noaugs], dim=0)
            if args.use_len_spd:
                batched_g.len_shortest_path = torch.cat([g_noaug.pair_len_shortest_path for g_noaug in g_noaugs], dim=0)
            if args.use_num_spd:
                batched_g.num_shortest_path = torch.cat([g_noaug.pair_num_shortest_path for g_noaug in g_noaugs], dim=0)
            if args.use_cnb_jac:
                batched_g.undir_jac = torch.cat([g_noaug.pair_undir_jac for g_noaug in g_noaugs], dim=0)
            if args.use_cnb_aa:
                batched_g.undir_aa = torch.cat([g_noaug.pair_undir_aa for g_noaug in g_noaugs], dim=0)
            if args.use_cnb_ra:
                batched_g.undir_ra = torch.cat([g_noaug.pair_undir_ra for g_noaug in g_noaugs], dim=0)
            if args.use_degree:
                batched_g.undir_degree = torch.cat([g_noaug.pair_undir_degree for g_noaug in g_noaugs], dim=0)
                if directed:
                    batched_g.in_degree = torch.cat([g_noaug.pair_in_degree for g_noaug in g_noaugs], dim=0)
                    batched_g.out_degree = torch.cat([g_noaug.pair_out_degree for g_noaug in g_noaugs], dim=0)

        return batched_g, z, x, edge_weights, y

    # pdb.set_trace()
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # True-----------------------
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )
    # pdb.set_trace()
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = GraphDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )
    final_val_loader = GraphDataLoader(
        final_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )
    final_test_loader = GraphDataLoader(
        final_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )


if args.train_node_embedding:
    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
elif args.pretrained_node_embedding:
    weight = torch.load(args.pretrained_node_embedding)
    emb = torch.nn.Embedding.from_pretrained(weight)
    emb.weight.requires_grad=False
else:
    emb = None

if 'DGCNN' in args.model:
    if args.ngnn_code:
        if 0 < args.sortpool_k <= 1:  # Transform percentile to number.
            if args.dataset.startswith("ogbl-citation"):
                # For this dataset, subgraphs extracted around positive edges are
                # rather larger than negative edges. Thus we sample from 1000
                # positive and 1000 negative edges to estimate the k (number of 
                # nodes to hold for each graph) used in SortPooling.
                # You can certainly set k manually, instead of estimating from
                # a percentage of sampled subgraphs.
                _sampled_indices = list(range(1000)) + list(
                    range(len(train_dataset) - 1000, len(train_dataset))
                )
            else:
                _sampled_indices = list(range(1000))
            # 加上 tqdm 进度条

            # 多线程提取节点数量
            def extract_num_nodes_parallel(train_dataset, indices):
                """
                使用多线程提取 train_dataset 中给定索引的节点数量，并转换为 NumPy 数组
                """
                num_nodes_list = []

                # 定义一个内部函数，用于多线程获取每个图的节点数量
                def get_num_nodes(index):
                    return train_dataset[index][0].num_nodes()

                # 使用 ThreadPoolExecutor 进行并行提取
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(get_num_nodes, i) for i in indices]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting num_nodes in parallel"):
                        num_nodes_list.append(future.result())

                # 转换为 NumPy 数组返回
                return np.array(num_nodes_list)

            # 使用 Numba 加速排序
            # @njit(parallel=True)
            def calculate_num_nodes_numba(num_nodes_array):
                """
                使用 Numba 并行加速排序节点数量数组
                """
                return np.sort(num_nodes_array)

            # 使用多线程提取节点数量，并转换为 NumPy 数组
            num_nodes_array = extract_num_nodes_parallel(train_dataset, _sampled_indices)

            # 使用 Numba 进行排序加速
            _num_nodes = calculate_num_nodes_numba(num_nodes_array)
            # print(_num_nodes)
            # _num_nodes = sorted(
            #     [train_dataset[i][0].num_nodes() for i in tqdm(_sampled_indices, desc="Calculating num_nodes")]
            # )
            _k = _num_nodes[int(math.ceil(args.sortpool_k * len(_num_nodes))) - 1]
            model_k = max(10, _k)
        else:
            model_k = int(args.sortpool_k)
    else:
        model_k = args.sortpool_k

print(f'args: {args}')
for run in range(args.runs):
    if args.model == 'DGCNN':
        if not args.ngnn_code:  # sieg_code
            print('sieg_code')
            model = DGCNN(args, args.hidden_channels, args.num_layers, args.max_z, model_k, 
                          train_dataset, use_feature=args.use_feature, 
                          node_embedding=emb).to(device)
        else:  # ngnn_code
            print('ngnn_code')
            model = ngnn_models.DGCNN(
                args.hidden_channels,
                args.num_layers,
                args.max_z,
                model_k,
                feature_dim=graph.ndata["feat"].size(1)
                if (args.use_feature and "feat" in graph.ndata)
                else 0,
                dropout=args.dropout,
                ngnn_type=args.ngnn_type,
                num_ngnn_layers=args.num_ngnn_layers,
            ).to(device)
    elif args.model == 'SAGE':
        model = SAGE(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                     args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GCN':
        model = GCN(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GIN':
        model = GIN(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GCNGraphormer':
        model = GCNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT, node_embedding=emb).to(device)
    elif args.model == 'GCNFFNGraphormer':
        model = GCNFFNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT, node_embedding=emb).to(device)
    elif args.model == 'GCNGraphormer_noNeigFeat':
        z_emb_agg = False if args.dont_z_emb_agg else True
        model = GCNGraphormer_noNeigFeat(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT, node_embedding=emb, z_emb_agg=z_emb_agg).to(device)
    elif args.model == 'SingleFFN':
        model = SingleFFN(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    use_feature=args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'FFNGraphormer':
        model = FFNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT, node_embedding=emb).to(device)
    elif args.model == 'DGCNNGraphormer':
        model = DGCNNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z,
                    k=model_k, train_dataset=train_dataset,
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT,
                    node_embedding=emb, readout_type=args.readout_type).to(device)
    elif args.model == 'DGCNNGraphormer_noNeigFeat':
        model = DGCNNGraphormer_noNeigFeat(args, args.hidden_channels, args.num_layers, args.max_z,
                    k=model_k, train_dataset=train_dataset,
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT,
                    node_embedding=emb, readout_type=args.readout_type).to(device)
    elif args.model == 'NGNNDGCNNGraphormer':
        model = NGNNDGCNNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z,
                k=model_k, feature_dim=graph.ndata["feat"].size(1),
                use_feature=args.use_feature, use_feature_GT=args.use_feature_GT,
                node_embedding=emb, readout_type=args.readout_type).to(device)
    elif args.model == 'DGCNN_noNeigFeat':
        model = DGCNN_noNeigFeat(args, args.hidden_channels, args.num_layers, args.max_z, model_k, 
                        train_dataset, use_feature=args.use_feature, 
                        node_embedding=emb).to(device)
    elif args.model == 'NGNNDGCNN_noNeigFeat':
        model = ngnn_models.DGCNN_noNeigFeat(
            args.hidden_channels,
            args.num_layers,
            args.max_z,
            model_k,
            feature_dim=graph.ndata["feat"].size(1)
            if (args.use_feature and "feat" in graph.ndata)
            else 0,
            dropout=args.dropout,
            ngnn_type=args.ngnn_type,
            num_ngnn_layers=args.num_ngnn_layers,
        ).to(device)
    elif args.model == 'NGNNDGCNNGraphormer_noNeigFeat':
        model = NGNNDGCNNGraphormer_noNeigFeat(args, args.hidden_channels, args.num_layers, args.max_z,
                k=model_k, feature_dim=graph.ndata["feat"].size(1),
                use_feature=args.use_feature, use_feature_GT=args.use_feature_GT,
                node_embedding=emb, readout_type=args.readout_type).to(device)
    elif args.model == 'SingleGraphormer':
        model = SingleGraphormer(args, args.hidden_channels, args.num_layers, args.max_z, 
                      train_dataset=train_dataset, use_feature=args.use_feature, use_feature_GT=args.use_feature_GT, 
                      node_embedding=emb).to(device)
    print(model)
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    lr = 2 * args.lr if args.scheduler else args.lr
    optimizer = torch.optim.Adam(params=parameters, lr=lr)  # , weight_decay=0.002
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)
    total_params = sum(p.numel() for param in parameters for p in param)
    localtime = time.asctime(time.localtime(time.time()))
    print(f'{localtime} Total number of parameters is {total_params}')
    # if args.model.find('DGCNN') != -1:
    #     print(f'SortPooling k is set to {model.k}')
    with open(log_file, 'a') as f:
        print(f'Total number of parameters is {total_params}', file=f)
        if args.model.find('DGCNN') != -1:
            print(f'SortPooling k is set to {model_k}', file=f)

    start_epoch = 1
    if args.continue_from is not None:
        model.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_model_checkpoint{}.pth'.format(args.continue_from[0], args.continue_from[-1])))
        )
        if not args.only_final_test:
            optimizer.load_state_dict(
                torch.load(os.path.join(args.res_dir, 
                    'run{}_optimizer_checkpoint{}.pth'.format(args.continue_from[0], args.continue_from[-1])))
            )
            start_epoch = args.continue_from[-1] + 1
            # args.epochs -= args.continue_from

    if args.part_continue_from is not None:  # stage2 training
        model.ngnndgcnn.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_model_checkpoint{}.pth'.format(args.part_continue_from[0], args.part_continue_from[-1])))
        )
        for p in model.ngnndgcnn.parameters(): p.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)
        start_epoch = 1001  # stage2的模型从1001开始

    if args.only_test:
        results = test(args.eval_metric)
        for key, result in results.items():
            loggers[key].add_result(run, result)
        for key, result in results.items():
            valid_res, test_res = result
            localtime = time.asctime(time.localtime(time.time()))
            print(f'[{localtime}] {key}')
            print(f'[{localtime}] Run: {run + 1:02d}, '
                  f'[{localtime}] Valid: {100 * valid_res:.2f}%, '
                  f'[{localtime}] Test: {100 * test_res:.2f}%')
        # pdb.set_trace()
        exit()

    if args.only_final_test:  # need continue_from
        results = final_test(args.eval_metric)
        for key in loggers.keys():
            result = results[key]
        final_valid_str = []
        final_test_str = []
        for key, result in results.items():
            final_valid_res, final_test_res = result
            final_valid_str.append(f'{key} {100 * final_valid_res:.2f}%')
            final_test_str.append(f'{key} {100 * final_test_res:.2f}%')

        to_print = (f'Run: {args.continue_from[0]:02d}, Epoch: {args.continue_from[-1]}, ' +
                    f'Final Valid: {", ".join(final_valid_str)}, ' +
                    f'Final Test: {", ".join(final_test_str)}')
        print(f'{to_print}')
        with open(log_file, 'a') as f:
            print(to_print, file=f)
        # pdb.set_trace()
        exit()

    if len(args.test_multiple_models) > 0 :
        model_paths = args.test_multiple_models
        num_models = len(model_paths)
        models = []
        for path in model_paths:
            m = cp.deepcopy(model)
            m.load_state_dict(torch.load(path))
            models.append(m)
        print(f'       tag - percent - mem.free(Gb) - mem.avail(Gb) - mem.used(Gb)')
        mem = psutil.virtual_memory()
        print(f'     begin - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
        if args.eval_metric != 'auc':
            val_preds, val_trues, pos_val_preds, neg_val_preds = test_multiple_models(models, val_loader, len(val_dataset))
        else:
            val_preds, val_trues, _, _ = test_multiple_models(models, val_loader, len(val_dataset))
        mem = psutil.virtual_memory()
        print(f' after val - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
        if args.eval_metric != 'auc':
            test_preds, test_trues, pos_test_preds, neg_test_preds = test_multiple_models(models, test_loader, len(test_dataset))
        else:
            test_preds, test_trues, _, _ = test_multiple_models(models, test_loader, len(test_dataset))
        mem = psutil.virtual_memory()
        print(f'after test - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
        results = eval_multiple_models(num_models,
                            pos_val_preds=pos_val_preds, neg_val_preds=neg_val_preds, pos_test_preds=pos_test_preds, neg_test_preds=neg_test_preds,
                            val_preds=val_preds, val_trues=val_trues, test_preds=test_preds, test_trues=test_trues, eval_metric=args.eval_metric)
        if args.eval_metric != 'auc':
            results_auc = eval_multiple_models(num_models,
                                pos_val_preds=pos_val_preds, neg_val_preds=neg_val_preds, pos_test_preds=pos_test_preds, neg_test_preds=neg_test_preds,
                                val_preds=val_preds, val_trues=val_trues, test_preds=test_preds, test_trues=test_trues, eval_metric='auc')
            for i in range(num_models):
                for key in results_auc[i].keys():
                    results[i][key] = results_auc[i][key]
        mem = psutil.virtual_memory()
        print(f'     final - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
        for i, path in enumerate(model_paths):
            print(path)
            with open(log_file, 'a') as f:
                print(path, file=f)
            for key in loggers.keys():
                result = results[i][key]
                loggers[key].add_result(run, result)
            valid_str = []
            test_str = []
            for key, result in results[i].items():
                valid_res, test_res = result
                valid_str.append(f'{key} {100 * valid_res:.2f}%')
                test_str.append(f'{key} {100 * test_res:.2f}%')
            to_print = (f'Run: {run + 1:02d}, ' +
                        f'Valid: {", ".join(valid_str)}, ' +
                        f'Test: {", ".join(test_str)}')
            localtime = time.asctime(time.localtime(time.time()))
            print(f'[{localtime}] {to_print}')
            with open(log_file, 'a') as f:
                print(to_print, file=f)
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run, std=True)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f, std=True)
        # pdb.set_trace()
        exit()

    # Training starts
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # import pdb; pdb.set_trace()
        loss, train_result = train(len(train_dataset))  # {'AUC': 0.9661961285501943}

        if epoch % args.eval_steps == 0:
            results = test(args.eval_metric)  # {'MRR': (0.7427010536193848, 0.7336037158966064), 'AUC': (0.9981022174148187, 0.9458885884261763)}
            for key in loggers.keys():  # MRR
                result = results[key]
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                model_name = os.path.join(
                    args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run+1, epoch))
                optimizer_name = os.path.join(
                    args.res_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run+1, epoch))
                torch.save(model.state_dict(), model_name)
                torch.save(optimizer.state_dict(), optimizer_name)

                train_str = []
                for key, result in train_result.items():
                    train_str.append(f'{key} {100 * result:.2f}%')  # ['AUC 73.11%']
                valid_str = []  # ['MRR 100.00%', 'AUC 100.00%']
                test_str = []
                for key, result in results.items():
                    valid_res, test_res = result
                    valid_str.append(f'{key} {100 * valid_res:.2f}%')
                    test_str.append(f'{key} {100 * test_res:.2f}%')

                # pdb.set_trace()
                to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                            f'Loss: {loss:.4f}, {", ".join(train_str)}, ' +
                            f'Valid: {", ".join(valid_str)}, ' +
                            f'Test: {", ".join(test_str)}')
                localtime = time.asctime(time.localtime(time.time()))
                print(f'[{localtime}] {to_print}')
                with open(log_file, 'a') as f:
                    print(to_print, file=f)

    # choose the best model in valid for final_valid and final_test
    if len(loggers.keys()) > 1:
        candidate_idxs, candidate_results = [], []
        for key in loggers.keys():
            res = torch.tensor([val_test[0] for val_test in loggers[key].results[run]])
            candidate_idxs += (torch.topk(res, 1, largest=True).indices).tolist()
        for key in loggers.keys():
            res = torch.tensor([val_test[0] for val_test in loggers[key].results[run]])
            candidate_results.append(res[candidate_idxs].tolist())
        candidate_results_mean = torch.mean(torch.Tensor(candidate_results), 0)
        idx_to_test = [(candidate_idxs[torch.topk(candidate_results_mean, 1, largest=True).indices] + 1)]
        if args.part_continue_from is not None: idx_to_test = [i+1000 for i in idx_to_test]
        candidate_epochs = [i+1 for i in candidate_idxs] if args.part_continue_from is None else [i+1001 for i in candidate_idxs]
        print(f'candidate_epochs: {candidate_epochs}\ncandidate_results: {candidate_results}\ncandidate_results_mean: {candidate_results_mean.tolist()}')
        with open(log_file, 'a') as f:
            print(f'candidate_epochs: {candidate_epochs}\ncandidate_results: {candidate_results}\ncandidate_results_mean: {candidate_results_mean.tolist()}', file=f)
    else:
        for key in loggers.keys():
            res = torch.tensor([val_test[0] for val_test in loggers[key].results[run]])
        idx_to_test = (
            torch.topk(res, 1, largest=True).indices + 1
        ).tolist()  # indices of top 1 valid results
        if args.part_continue_from is not None: idx_to_test = [i+1000 for i in idx_to_test]


    for _idx, epoch in enumerate(idx_to_test):
        model_name = os.path.join(
            args.res_dir, f"run{run+1}_model_checkpoint{epoch}.pth"
        )
        optimizer_name = os.path.join(
            args.res_dir,
            f"run{run+1}_optimizer_checkpoint{epoch}.pth",
        )
        model.load_state_dict(torch.load(model_name))
        optimizer.load_state_dict(torch.load(optimizer_name))

        results = final_test(args.eval_metric)
        for key in loggers.keys():
            result = results[key]
            loggers[key].add_result(run, result)

            final_valid_str = []
            final_test_str = []
            for key, result in results.items():
                final_valid_res, final_test_res = result
                final_valid_str.append(f'{key} {100 * final_valid_res:.2f}%')
                final_test_str.append(f'{key} {100 * final_test_res:.2f}%')

            to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                        f'Final Valid: {", ".join(final_valid_str)}, ' +
                        f'Final Test: {", ".join(final_test_str)}')
            localtime = time.asctime(time.localtime(time.time()))
            print(f'[{localtime}] {to_print}')
            with open(log_file, 'a') as f:
                print(to_print, file=f)


    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(run)
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(run, f=f)
        print(f'runs 0-{run}')
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(f'runs 0-{run}', file=f)
            loggers[key].print_statistics(f=f)

for key in loggers.keys():
    print(key)
    loggers[key].print_statistics()
    with open(log_file, 'a') as f:
        print(key, file=f)
        loggers[key].print_statistics(f=f)
print(f'Total number of parameters is {total_params}')
print(f'Results are saved in {args.res_dir}')


