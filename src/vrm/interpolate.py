import torch
import torch.nn.functional as F
import random

import json
from torch_geometric.data import Data
import pickle
from transformers import AutoTokenizer, AutoModel

from utils import seed_everything, index_to_mask
from edge.add_edge import *
from edge.pretrain import *

def data_augment(data, k, args):
    data = data.to(args.device)
    seed_everything(args.seed)

    imb_class = args.imb_class
    maj_num, min_num = max(data.class_num), min(data.class_num)

    train_idxs = data.train_mask.nonzero().view(-1)
    val_idxs = data.val_mask.nonzero().view(-1)
    test_idxs = data.test_mask.nonzero().view(-1)
    data.old_train_size = train_idxs.shape[0]

   
    
    if args.llm == "True":
        new_xs, new_ys, new_edge_index, num_new_nodes = interpolate_llm(data, args, imb_class, maj_num, train_idxs, k)
    else:
        new_xs, new_ys,  new_edge_index, num_new_nodes = interpolate_numeric(data, args, imb_class, maj_num, train_idxs, k)

    
    if args.save:
        # Define output path (you can customize the folder or filename)
        output_dir = "augmented_features"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{args.dataset}_{args.method}_{'llm' if args.llm=='True' else 'nonllm'}.pt"
        save_path = os.path.join(output_dir, filename)

        torch.save({'new_emb': new_xs,'new_labels': new_ys,}, save_path)
        print(f"Saved new_xs tensor to: {save_path}")


    data.x = torch.cat((data.x, new_xs.to(args.device)), 0)
    data.y = torch.cat((data.y, new_ys), 0)
    data.num_nodes = data.x.shape[0]

    # Edge generation
    data= integrate_new_edges(data, new_xs, num_new_nodes, args,new_edge_index)
    
    # Update masks
    new_idxs = torch.arange(data.num_nodes - new_xs.shape[0], data.num_nodes).to(args.device)
    new_train_idxs = torch.cat((train_idxs, new_idxs), 0)

    data.train_mask = index_to_mask(new_train_idxs, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idxs, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idxs, size=data.num_nodes)

    # Recalculate class distribution
    data.class_num = [(data.y == i).logical_and(data.train_mask).sum().item() for i in data.y.unique()]
    print("Class number after data augmentation:", data.class_num)

    return data



def interpolate_llm(data, args, imb_class, maj_num, train_idxs, k):
    """
    Generate new examples via LLM prompts + embedding.

    Returns:
      new_xs: Tensor of shape (N_new, d)
      new_ys: Tensor of shape (N_new,)
      new_edge_tensor: LongTensor of shape (2, E_new)
    """
    new_xs, new_ys = [], []
    new_edge_index = [[], []]
    num_new_nodes = 0
    prompt_list = []

    for i in range(len(imb_class)):  # for each minority class
        
        start_category = data.categories[imb_class[i]]

        idxs = ((data.y == imb_class[i]) & data.train_mask).nonzero().view(-1)
        x = data.x[idxs]
        count = maj_num - len(idxs)

        if args.method =="S":
            prompt_list, new_ys, new_edge_index, num_new_nodes=llm_itp_S(x, idxs, k, count, data, imb_class, i, args, new_ys, new_edge_index, num_new_nodes, start_category, prompt_list)
            
        elif args.method == "M":
            prompt_list, new_ys, new_edge_index, num_new_nodes=llm_itp_M(x, idxs, k, count, data, train_idxs, imb_class, i, args, new_ys, new_edge_index, num_new_nodes, start_category, prompt_list)
         

        elif args.method in ["O", "zero_shot"]: 
            prompt_list, new_ys, new_edge_index, num_new_nodes=llm_itp_O(x, idxs, count, data, imb_class, i, args, new_ys, new_edge_index, num_new_nodes, start_category, prompt_list)
        
                    
        elif args.method == "few_shots":
            prompt_list, new_ys, new_edge_index, num_new_nodes=llm_few_shot(x, idxs, count, data, imb_class, i, args, new_ys, new_edge_index, num_new_nodes, start_category, prompt_list)
          

    # Handle LLM-generated data
    json_name = f"llm_response/{args.method}_{args.dataset}{args.model_name}"
    output_json_path = f"{json_name}_{args.imb_ratio}.json" if args.imb_ratio != 0.2 else f"{json_name}.json"

    if args.load_response:
        with open(output_json_path) as file:
            loaded_data = json.load(file)
        answer_list = [obj["answer"] for obj in loaded_data]
    else:
        batch_size = {"Photo": 2, "Computer": 2, "Child": 1}.get(args.dataset, 4)
        print("Batch Size for text generation:", batch_size)
        answer_list, output_list = process_prompt_list(prompt_list, args, batch_size=batch_size)

        with open(output_json_path, "w") as outfile:
            json.dump(output_list, outfile, indent=4)

    print("Using", args.embed)
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    new_xs = torch.tensor(encoder.encode(answer_list))
    new_ys = torch.tensor(new_ys).to(args.device)
    
    return new_xs, new_ys, new_edge_index, num_new_nodes


def interpolate_numeric(data, args, imb_class, maj_num, train_idxs, k):
    """
    Generate new examples by numeric interpolation.

    Returns:
      new_xs: Tensor of shape (N_new, d)
      new_ys: Tensor of shape (N_new,)
      new_edge_tensor: LongTensor of shape (2, E_new)
    """

    new_xs, new_ys = [], []
    new_edge_index = [[], []]
    num_new_nodes = 0
    
    for i in range(len(imb_class)):  # for each minority class
        idxs = ((data.y == imb_class[i]) & data.train_mask).nonzero().view(-1)
        x = data.x[idxs]
        count = maj_num - len(idxs)

        if args.method =="S":
            new_xs, new_ys, new_edge_index, num_new_nodes= num_itp_S(x, idxs, count, k, imb_class, i, data, args, new_xs, new_ys, new_edge_index, num_new_nodes)
            
        elif args.method == "M": 
            new_xs, new_ys, new_edge_index, num_new_nodes= num_itp_M(data, x, idxs, train_idxs, count, k, imb_class, i, args, new_xs, new_ys, new_edge_index, num_new_nodes)
            
        if args.method == "O":   
            new_xs, new_ys, new_edge_index, num_new_nodes = num_itp_O(x, idxs, count, imb_class, i, args, new_xs, new_ys, new_edge_index, data, num_new_nodes)

                

    new_xs = torch.stack(new_xs).to(args.device)
    new_ys = torch.tensor(new_ys).to(args.device)

    return new_xs, new_ys, new_edge_index, num_new_nodes









def integrate_new_edges(data,new_xs,num_new_nodes,args,new_edge_index):
 
        
   
    if args.CF == "True":
        
        
        
        edge_list = []
        # for each new node, connect it to all existing nodes up to that point
        for c in range(num_new_nodes):
            # cur_count = index of the new node
            cur_count = c + data.x.shape[0] - new_xs.shape[0]
            # build a complete bipartite connection: new node -> all existing
            edge = torch.zeros((2, cur_count), dtype=torch.long)
            edge[0] = cur_count
            edge[1] = torch.arange(cur_count)
            edge_list.append(edge)
        edges = torch.cat(edge_list, dim=1)
        
        
        # load pretrained condidence function
        
        encoder, predictor=load_model(data, args)
        # confidence_function should return a [2, E] tensor of new edges
        new_edge_index = confidence_function(data, num_new_nodes, edges,encoder, predictor, args)

    else:
        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long)
        # shift all source indices by the number of original nodes
        offset = data.x.shape[0] - new_xs.shape[0]
        new_edge_index[0] += offset

    # append to existing edges
    data.edge_index = torch.cat(
        (data.edge_index, new_edge_index.to(args.device)), dim=1
    )
    return data





def num_itp_S(x, idxs, count, k, imb_class, i, data, args, new_xs, new_ys, new_edge_index, num_new_nodes):
    norm_x = F.normalize(x, p=2, dim=1)
    cosine_similarity = torch.mm(norm_x, norm_x.t())
    _, top_indices = torch.topk(
        cosine_similarity, k=min(k + 1, cosine_similarity.shape[1]), dim=1
    )
    top_indices = top_indices[:, 1:]  # exclude self
    start_idx = [0 for _ in range(top_indices.shape[0])]
    
    while count != 0:
        for j in range(len(idxs)):
            ori_idx_start = idxs[j]
            start_x = x[j]

            idx = top_indices[j][start_idx[j]]
            ori_idx_end = idxs[idx]
            end_x = x[idx]

            k = min(k, cosine_similarity.shape[1] - 1)
            start_idx[j] = (start_idx[j] + 1) % k

            alpha = random.random()
            new_x = start_x + alpha * (end_x - start_x)
            new_xs.append(new_x)
            new_ys.append(imb_class[i])
            
            if args.CF == "False":
                new_edge_index = edge_duplication(data, new_edge_index, ori_idx_start, num_new_nodes)
    

            num_new_nodes += 1
            count -= 1
            if count == 0:
                break

            
    return new_xs, new_ys, new_edge_index, num_new_nodes


def num_itp_M(data, x, idxs, train_idxs, count, k, imb_class, i, args, new_xs, new_ys, new_edge_index, num_new_nodes):
    train_x = data.x[train_idxs]
    train_y = data.y[train_idxs]
    
    norm_x = F.normalize(x, p=2, dim=1)
    norm_train_x = F.normalize(train_x, p=2, dim=1)
    cosine_similarity = torch.mm(norm_x, train_x.t())

    _, top_indices = torch.topk(
        cosine_similarity, k=min(k + 1, cosine_similarity.shape[1]), dim=1
    )
    top_indices = top_indices[:, 1:]  # exclude self
    start_idx = [0 for _ in range(top_indices.shape[0])]
    
    while count != 0:
        for j in range(len(idxs)):
            ori_idx_start = idxs[j]
            start_x = x[j]

            idx = top_indices[j][start_idx[j]]

            ori_idx_end = train_idxs[idx]
            end_x = train_x[idx]
            end_y = train_y[idx]

            k = min(k, cosine_similarity.shape[1] - 1)
            start_idx[j] = (start_idx[j] + 1) % k

            alpha = random.random()
            new_x = start_x + alpha * (end_x - start_x)
            new_xs.append(new_x)
            new_ys.append(imb_class[i])
            
            if args.CF == "False":
                new_edge_index = edge_duplication(data, new_edge_index, ori_idx_start, num_new_nodes)

            num_new_nodes += 1
            count -= 1
            if count == 0:
                break

    return new_xs, new_ys, new_edge_index, num_new_nodes




def num_itp_O(x, idxs, count, imb_class, i, args, new_xs, new_ys, new_edge_index, data, num_new_nodes):
    while count != 0:
        for j in range(len(idxs)):
            ori_idx_start = idxs[j]
            start_x = x[j]
            new_x = start_x
            new_xs.append(new_x)
            new_ys.append(imb_class[i])
            
            if args.CF == "False":
                new_edge_index = edge_duplication(data, new_edge_index, ori_idx_start, num_new_nodes)

            num_new_nodes += 1
            count -= 1
            if count == 0:
                break

    return new_xs, new_ys, new_edge_index, num_new_nodes



def llm_itp_S(x, idxs, k, count, data, imb_class, i, args, new_ys, new_edge_index, num_new_nodes, start_category, prompt_list):
    norm_x = F.normalize(x, p=2, dim=1)
    cosine_similarity = torch.mm(norm_x, norm_x.t())
    _, top_indices = torch.topk(
        cosine_similarity, k=min(k + 1, cosine_similarity.shape[1]), dim=1
    )
    top_indices = top_indices[:, 1:]  # exclude self
    start_idx = [0 for _ in range(top_indices.shape[0])]

    while count != 0:
        for j in range(len(idxs)):
            ori_idx_start = idxs[j]
            start_x = x[j]
            idx = top_indices[j][start_idx[j]]

            ori_idx_end = idxs[idx]
            end_x = x[idx]

            k = min(k, cosine_similarity.shape[1] - 1)
            start_idx[j] = (start_idx[j] + 1) % k
            start_text = data.raw_text[ori_idx_start]

            end_text = data.raw_text[ori_idx_end]

            prompt = form_prompt(start_text, end_text, start_category, start_category, args)
            prompt_list.append(prompt)

            new_ys.append(imb_class[i])

            if args.CF == "False":
                new_edge_index = edge_duplication(data, new_edge_index, ori_idx_start, num_new_nodes)

            num_new_nodes += 1
            count -= 1
            if count == 0:
                break

    return prompt_list, new_ys, new_edge_index, num_new_nodes




def llm_itp_M(x, idxs, k, count, data, train_idxs, imb_class, i, args, new_ys, new_edge_index, num_new_nodes, start_category, prompt_list):
    norm_x = F.normalize(x, p=2, dim=1)
    train_x = data.x[train_idxs]
    train_y = data.y[train_idxs]

    norm_train_x = F.normalize(train_x, p=2, dim=1)
    cosine_similarity = torch.mm(norm_x, train_x.t())

    _, top_indices = torch.topk(
        cosine_similarity, k=min(k + 1, cosine_similarity.shape[1]), dim=1
    )
    top_indices = top_indices[:, 1:]  # exclude self
    start_idx = [0 for _ in range(top_indices.shape[0])]

    while count != 0:
        for j in range(len(idxs)):
            ori_idx_start = idxs[j]
            start_x = x[j]

            idx = top_indices[j][start_idx[j]]

            ori_idx_end = train_idxs[idx]
            end_x = train_x[idx]
            end_y = train_y[idx]

            k = min(k, cosine_similarity.shape[1] - 1)
            start_idx[j] = (start_idx[j] + 1) % k

            start_text = data.raw_text[ori_idx_start]
            end_text = data.raw_text[ori_idx_end]
            end_category = data.categories[end_y]
            prompt = form_prompt(start_text, end_text, start_category, end_category, args)

            prompt_list.append(prompt)
            new_ys.append(imb_class[i])

            if args.CF == "False":
                new_edge_index = edge_duplication(data, new_edge_index, ori_idx_start, num_new_nodes)

            num_new_nodes += 1
            count -= 1
            if count == 0:
                break

    return prompt_list, new_ys, new_edge_index, num_new_nodes


def llm_itp_O(x, idxs, count, data, imb_class, i, args, new_ys, new_edge_index, num_new_nodes, start_category, prompt_list):
    while count != 0:
        for j in range(len(idxs)):
            ori_idx_start = idxs[j]
            start_x = x[j]
            start_text = data.raw_text[ori_idx_start]

            prompt = form_prompt(start_text, "", start_category, "", args)
            prompt_list.append(prompt)

            new_ys.append(imb_class[i])

            if args.CF == "False":
                new_edge_index = edge_duplication(data, new_edge_index, ori_idx_start, num_new_nodes)

            num_new_nodes += 1
            count -= 1
            if count == 0:
                break

    return prompt_list, new_ys, new_edge_index, num_new_nodes


def llm_few_shot(x, idxs, count, data, imb_class, i, args, new_ys, new_edge_index, num_new_nodes, start_category, prompt_list):
    while count != 0:
        for j in range(len(idxs)):
            ori_idx_start = idxs[j]
            start_x = x[j]

            idx = torch.randint(0, len(idxs), (1,)).item()
            ori_idx_end = idxs[idx]
            end_x = x[idx]

            start_text = data.raw_text[ori_idx_start]
            end_text = data.raw_text[ori_idx_end]

            prompt = form_prompt(start_text, end_text, start_category, start_category, args)
            prompt_list.append(prompt)

            new_ys.append(imb_class[i])

            if args.CF == "False":
                new_edge_index = edge_duplication(data, new_edge_index, ori_idx_start, num_new_nodes)

            num_new_nodes += 1
            count -= 1
            if count == 0:
                break

    return prompt_list, new_ys, new_edge_index, num_new_nodes



