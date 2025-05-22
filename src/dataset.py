import os.path as osp
import torch_geometric.transforms as T
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, add_self_loops
import numpy as np
import warnings
import pickle


from utils import seed_everything, index_to_mask, model_encode, normalize_edge







warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore")

def get_dataset(name,  args):
    data = torch.load(f'./dataset/{name}.pt')
    seed_everything(args.seed) 

    my_num_val=500

    if name =='Cora':
        args.imb_class=list(range(5))
    elif name=='PubMed':
        args.imb_class=list(range(2))
    
    else:
        data.raw_text = data.raw_texts
       
        if name=="Photo":
            args.imb_class=list(range(8))
            my_num_val=200
        elif name=="Computer":
            args.imb_class=list(range(6))
            my_num_val=100
        elif name=="Child":
            args.imb_class=list(range(15))
            my_num_val=75
        elif name=="Citeseer":
            args.imb_class=list(range(4))
            # args.imb_class=list(range(2))
            my_num_val=150
     
            
    print("Imbalanced classes: ", args.imb_class)
            
    data = random_planetoid_splits(data, args.imb_class, args.imb_ratio,num_val=my_num_val)
        
       
        
    data.class_num = []
    for i in data.y.unique():
        data.class_num.append(torch.sum((data.y == i) & data.train_mask).item())
        
    print("Class number before data augmentation: ",data.class_num)

 


    if args.dataset=="Cora":

        data.categories=['Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods']
    elif args.dataset=="PubMed":
        data.categories=['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
    
    print(args.embed)
    
   
    
    # Text embeddings from sentence
    if args.embed=="SBERT" : 
        text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        data.x = torch.tensor(text_encoder.encode(data.raw_text), dtype=torch.float)
    else:
        if args.embed=="SimCSE":
            model_name="princeton-nlp/unsup-simcse-bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(  model_name)
            model = AutoModel.from_pretrained(  model_name)
            
        elif args.embed=="Llama3":
            model_name="meta-llama/Llama-3.2-1B"
            tokenizer = AutoTokenizer.from_pretrained(  model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained( model_name)
            
        
        
        data.x = torch.tensor(model_encode(data.raw_text,model, tokenizer ), dtype=torch.float)
    

    if args.method!="none" and not args.pretrain: 
        from vrm.interpolate import data_augment
        data = data_augment(data, 3,  args)
        
        if args.save:
            # We will save only the TRAINING subset to a file
            train_emb = data.x[data.train_mask]
            train_labels = data.y[data.train_mask]
            
            torch.save(
                {'train_emb': train_emb, 'train_labels': train_labels}, 
                f'./plot/{args.dataset}_{args.method}_{"llm" if args.llm == "True" else "nonllm"}.pt'
            )


    else:
        pass
    
 
    
   

    args.num_features = data.x.shape[1]
    args.num_nodes = data.x.shape[0]
    args.num_classes = data.y.max().item() + 1
    
    
    data.edge_index, _ = add_remaining_self_loops(
        data.edge_index, num_nodes=data.num_nodes) # add remaining self-loops (i,i) for every node i in  the graoh
    data.edge_index = to_undirected(data.edge_index, data.num_nodes) # if (i,j) in edge_index, generate (j,i) in edge_index
    

    # calculate the degree normalize term: the product of the inverse square roots of the degrees of the source and destination nodes
    row, col = data.edge_index # source and destination nodes
    deg = degree(col, data.num_nodes)
    deg_inv_sqrt = deg.pow(-0.5) 

    
    data.edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value = data.edge_weight, is_sorted=False)
    

    #for link prediction
    transform = T.RandomLinkSplit(is_undirected=True, neg_sampling_ratio = 1.0, num_val = 0.1, num_test = 0.2) # negative sampling
    train_data, val_data, test_data = transform(data)
    train_edge, val_edge, test_edge = train_data.edge_label_index, val_data.edge_label_index, test_data.edge_label_index
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = train_edge[:, :train_edge.shape[1]//2].t()
    split_edge['train']['edge_neg'] = train_edge[:, train_edge.shape[1]//2:].t()
    split_edge['valid']['edge'] = val_edge[:, :val_edge.shape[1]//2].t()
    split_edge['valid']['edge_neg'] = val_edge[:, val_edge.shape[1]//2:].t()
    split_edge['test']['edge'] = test_edge[:, :test_edge.shape[1]//2].t()
    split_edge['test']['edge_neg'] = test_edge[:, test_edge.shape[1]//2:].t()


    data.train_edge_index, data.train_edge_weight_gcn, data.train_edge_weight_sage = normalize_edge(split_edge['train']['edge'], data.x.shape[0])
    data.train_adj_gcn = SparseTensor(row=data.train_edge_index[0], col=data.train_edge_index[1], value = data.train_edge_weight_gcn, is_sorted=False)
    
    

   
    return data, split_edge, args



def random_planetoid_splits(data, imb_class, imb_ratio, num_train_node=20,num_val=500):
    indices = []
    
    num_classes = data.y.max().item() + 1
    for i in range(num_classes):
        index = torch.nonzero(data.y == i).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    #print(indices)
    train_index = []
    res_index = []
    for i, _ in enumerate(indices):
        #print(i,_)
        if i not in imb_class:
            train_index.append(_[:num_train_node])
            res_index.append(_[num_train_node:])
        else:
            train_index.append(_[:int(num_train_node*imb_ratio)])
            res_index.append(_[int(num_train_node*imb_ratio):])
    
    train_index = torch.cat(train_index, dim=0)
    res_index = torch.cat(res_index, dim=0)
    
    res_index = res_index[torch.randperm(res_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(res_index[:num_val], size=data.num_nodes)
    data.test_mask = index_to_mask(res_index[num_val:], size=data.num_nodes)

    return data


