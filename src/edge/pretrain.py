# import argparse
from os import path
from tqdm import tqdm
import random
from torch import tensor
from torch_scatter import scatter_max, scatter_add
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from parser import args
from dataset import get_dataset
from gnn.learn import *
from gnn.model import *
from utils import *
    
    
def train_cf(args):
    print("Training confidence function...")
    data, split_edge, args = get_dataset(args.dataset, args)

    data = data.to(args.device)

    train_hits, val_hits, test_hits = [], [], []

    res = {}


    train_hits, val_hits, test_hits = [], [], []
    # for _ in pbar:
        
    encoder = MLP(data.x.shape[1], args.n_hidden, args.n_hidden, args.n_layers, args.en_dp).to(args.device)


    predictor = MLP_predictor(args.n_hidden, args.n_hidden, 1, args.n_layers, args.lp_dp).to(args.device)
    optimizer = torch.optim.Adam([{'params': encoder.parameters(), "lr": args.encoder_lr},
                                    {'params': predictor.parameters(), 'lr': args.predictor_lr}])

    best_val_hit = -math.inf
    
    for epoch in range(0, args.epochs):
        loss = train_lp(encoder, predictor, data, optimizer, split_edge['train']['edge'], args)
        ress=eval_lp(encoder, predictor, data, split_edge, args)

        if epoch%100==0:
            print("Epoch: ",epoch)
        if ress['valid'][args.track_idx] > best_val_hit:
            best_val_hit = ress['valid'][args.track_idx]
            ress_final = ress 
            
    model_dir = f'CF/{args.dataset}_CF/'
    os.makedirs(model_dir, exist_ok=True)
    encoder_path = os.path.join(model_dir, 'MLP_encoder.pt')
    predictor_path = os.path.join(model_dir, 'MLP_predictor.pt')

    
    # Save the models
    save_model(encoder, predictor, encoder_path, predictor_path)
    

    train_hits.append(ress_final['train'])
    val_hits.append(ress_final['valid'])
    test_hits.append(ress_final['test'])


    res["MLP"] = (np.mean(test_hits, axis = 0)[-1])

    print(res)
    print("Finishes pretraining.")




def save_model(encoder, predictor, encoder_path, predictor_path):
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(predictor.state_dict(), predictor_path)
    
    
def load_model(data, args):
    seed_everything(args.seed)
    
    dataset_dir=os.path.join("CF",f'{args.dataset}_CF')
    encoder_path = os.path.join(dataset_dir, 'MLP_encoder.pt')
    predictor_path = os.path.join(dataset_dir, 'MLP_predictor.pt')
    

    encoder = MLP(data.x.shape[1], args.n_hidden, args.n_hidden, args.n_layers, args.en_dp).to(args.device)
    predictor = MLP_predictor(args.n_hidden, args.n_hidden, 1, args.n_layers, args.lp_dp).to(args.device)
            
    encoder.load_state_dict(torch.load(encoder_path))
    predictor.load_state_dict(torch.load(predictor_path))

    return encoder, predictor