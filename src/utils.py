from torch_scatter import scatter
import torch
import numpy as np
from torch_geometric.utils import to_undirected,add_self_loops,  add_remaining_self_loops, k_hop_subgraph, degree
import random
from torch_scatter import scatter_add
import os
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

# def index_to_mask(train_index, val_index, test_index, size):
#     train_mask = torch.zeros(size, dtype=torch.bool)
#     val_mask = torch.zeros(size, dtype=torch.bool)
#     test_mask = torch.zeros(size, dtype=torch.bool)

#     train_mask[train_index] = 1
#     val_mask[val_index] = 1
#     test_mask[test_index] = 1

#     return train_mask, val_mask, test_mask



def model_encode(sentences, model, tokenizer):
    batch_size = 1
    embeddings = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    i = 0
    while i < len(sentences):
        # Prepare batch data
        if i + batch_size > len(sentences):
            batch_data = sentences[i:]
        else:
            batch_data = sentences[i:i + batch_size]

        # Tokenize batch data
        inputs = tokenizer(batch_data, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings from model
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        # Pooling method: Mean pooling over the last hidden layer tokens
        batch_embed = outputs.hidden_states[-1].mean(dim=1)
        
        # Store embeddings
        embeddings.append(batch_embed)

        # Move to the next batch
        i += batch_size

    # Concatenate all batch embeddings into a single tensor
    embeddings = torch.cat(embeddings, dim=0)

    return embeddings

def normalize_edge(edge_index, n_node):
    edge_index = to_undirected(edge_index.t(), num_nodes=n_node)

    edge_index, _ = add_self_loops(edge_index, num_nodes=n_node)

    row, col = edge_index
    deg = degree(col)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.

    edge_weight_gcn = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    edge_weight_sage = deg_inv_sqrt[row] * deg_inv_sqrt[row]

    return edge_index, edge_weight_gcn, edge_weight_sage