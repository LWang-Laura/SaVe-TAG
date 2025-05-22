from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor
import torch
import os 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gnn.model import *
from gnn.learn import *
from vrm.prompt import *
from utils import normalize_edge
from edge.pretrain import load_model

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


        


@torch.no_grad()
def confidence_function( data, num_added,edges,encoder, predictor,args):

    data = data.to(args.device)
    encoder.eval()
    predictor.eval()
    pos_edges= edges.t()
    h = encoder(data.x, edges)
   
    
    topk_pos_edges = []
    

    pos_preds = torch.cat([predictor(h[pos_edges[batch, 0]]*h[pos_edges[batch, 1]]).squeeze().cpu() \
        for batch in PermIterator(pos_edges.device, pos_edges.shape[0], args.eval_bsz, training = False)])
    
   
    topk_pos_indices = torch.topk(pos_preds, k=num_added*20, largest=True, sorted=True).indices
    topk_pos_edges = pos_edges[topk_pos_indices]
    
    topk_edges= torch.t(topk_pos_edges)
   
    return  topk_edges





def edge_duplication(data,new_edge_index, ori_idx_start, num_new_nodes):
    # Generate a new edge index by connecting the new node to the original neighbors.
    neighbors = data.edge_index[1][data.edge_index[0] == ori_idx_start.item()].tolist()
    new_edge_index[0].extend([num_new_nodes] * len(neighbors))
    new_edge_index[1].extend(neighbors)
    
    return new_edge_index