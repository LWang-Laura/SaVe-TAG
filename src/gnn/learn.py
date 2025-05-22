import torch.nn.functional as F
import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from imblearn.metrics import geometric_mean_score

from utils import *



class PermIterator:
    '''
    Iterator of a permutation
    Generate batches of indices for data permutation, used for training and evaluation tasks.
    '''
    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret

    
def train(model, data, optimizer, args):
    
    model.train()
    optimizer.zero_grad()

    embed = model(data.x, data.edge_index)
    logits = F.log_softmax(embed, dim=1)

    loss = {}
    loss['train'] = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss['val'] = F.nll_loss(logits[data.val_mask], data.y[data.val_mask])
    loss['train'].backward()
    optimizer.step()

    return loss


def evaluate(model, data, args):
    model.eval()

    with torch.no_grad():
        embed = model(data.x, data.edge_index)

        logits = F.log_softmax(embed, dim=1)

    evals = {}

    pred_val = logits[data.val_mask].max(1)[1]
   

    evals['val'] = pred_val.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum()
    
 

    tmp = confusion_matrix(data.y[data.val_mask].cpu().numpy(), pred_val.cpu().numpy())
    evals['val_per_class'] = tmp.diagonal() / tmp.sum(axis = 1)

    pred_test = logits[data.test_mask].max(1)[1]

    # Overall Accuracy.
    evals['test'] = pred_test.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum()
    
    # Accuracy per class.
    tmp = confusion_matrix(data.y[data.test_mask].cpu().numpy(), pred_test.cpu().numpy())
    evals['test_per_class'] = tmp.diagonal() / tmp.sum(axis = 1)

    y_true=data.y[data.test_mask].cpu().numpy()
    y_pred= pred_test.cpu().numpy()

    f1_macro = f1_score(y_true, y_pred, average='macro')

    evals['f1_macro']=f1_macro
    
    # Balanced Accuracy
    evals['bacc'] = balanced_accuracy_score(y_true, y_pred)

    # G-Mean
    evals['gmean'] = geometric_mean_score(y_true, y_pred, average='macro')


    return evals





def train_lp(encoder, predictor, data, optimizer, train_edge, args):
    encoder.train()
    predictor.train()

    neg_edge = negative_sampling(data.train_edge_index, num_neg_samples = train_edge.shape[0]).t()

    total_loss, count = 0, 0
    for batch in PermIterator(train_edge.device, train_edge.shape[0], args.train_bsz):
        # 'h' will be a tensor containing the node embeddings produced by the GNN
        h = encoder(data.x, data.train_adj_gcn) 
        pos_score = predictor(h[train_edge[batch, 0]]*h[train_edge[batch, 1]])
        pos_loss = -F.logsigmoid(pos_score).mean()

        neg_score = predictor(h[neg_edge[batch, 0]]*h[neg_edge[batch, 1]])
        neg_loss = -F.logsigmoid(-neg_score).mean()

        loss = (pos_loss + neg_loss) / 2

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * train_edge.shape[0]
        count += train_edge.shape[0]

    return total_loss / count


# Below functions are adopted from https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py
def eval(input_dict, K):
    y_pred_pos, y_pred_neg, type_info = _parse_and_check_input(input_dict)
    return _eval_hits(y_pred_pos, y_pred_neg, type_info, K)

def _parse_and_check_input(input_dict):
    y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']
    
    type_info = None

    # check the raw tyep of y_pred_pos
    if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
        raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

    # check the raw type of y_pred_neg
    if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
        raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

    # if either y_pred_pos or y_pred_neg is torch tensor, use torch tensor
    if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
        # converting to torch.Tensor to numpy on cpu
        if isinstance(y_pred_pos, np.ndarray):
            y_pred_pos = torch.from_numpy(y_pred_pos)

        if isinstance(y_pred_neg, np.ndarray):
            y_pred_neg = torch.from_numpy(y_pred_neg)

        # put both y_pred_pos and y_pred_neg on the same device
        y_pred_pos = y_pred_pos.to(y_pred_neg.device)

        type_info = 'torch'

    else:
        # both y_pred_pos and y_pred_neg are numpy ndarray
        type_info = 'numpy'
    
    return y_pred_pos, y_pred_neg, type_info
    
    

def _eval_hits(y_pred_pos, y_pred_neg, type_info, K):
    '''
        compute Hits@K
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    '''

    if len(y_pred_neg) < K:
        return {'hits@{}'.format(K): 1.}

    if type_info == 'torch':
        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    # type_info is numpy
    else:
        kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
        hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

    return {'hits@{}'.format(K): hitsK}


@torch.no_grad()
def eval_lp(encoder, predictor, data, split_edge, args):
    encoder.eval()
    predictor.eval()

    ress = {'train': [],
            'valid': [],
            'test': []}

    h = encoder(data.x, data.train_adj_gcn)

    # eval_per_edge
    for key in split_edge:
        edge, neg_edge = split_edge[key]['edge'], split_edge[key]['edge_neg']

        pos_preds = torch.cat([predictor(h[edge[batch, 0]]*h[edge[batch, 1]]).squeeze().cpu() \
            for batch in PermIterator(edge.device, edge.shape[0], args.eval_bsz, training = False)])
        
        # Concatenate a list of tensors containing prediction scores for each negative edge
        neg_preds = torch.cat([predictor(h[neg_edge[batch, 0]]*h[neg_edge[batch, 1]]).squeeze().cpu() \
            for batch in PermIterator(neg_edge.device, neg_edge.shape[0], args.eval_bsz, training = False)])


        for K in [5, 10, 20, 50, 100]:

            hits = eval({'y_pred_pos': pos_preds, 'y_pred_neg': neg_preds}, K)[f'hits@{K}']

            ress[key].append(hits)

    return ress