from tqdm import tqdm
from torch import tensor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig

from parser import args
from dataset import *
from gnn.learn import *
from gnn.model import *
from utils import *
from edge.pretrain import *



def run(data, args):
    
    data = data.to(args.device)

    
    accs = {}
    accs_per_class = {}
    f1s_macro={}
    
    gmeans = {}
    baccs = {}

    args.num_classes=int(args.num_classes)
    for args.model in ['GCN', 'SAGE', 'MLP']:
        
        accs[args.model] = np.zeros(args.runs)
        accs_per_class[args.model] = np.zeros((args.runs, args.num_classes))
        f1s_macro[args.model]=np.zeros(args.runs)
        gmeans[args.model] = np.zeros(args.runs)
        baccs[args.model] = np.zeros(args.runs)

        if args.model == 'GCN':
            model = GCN(data.x.shape[1], args.hidden, args.num_classes, args.layers, args.dropout).to(args.device)
        elif args.model == 'SAGE':
            model = SAGE(data.x.shape[1], args.hidden, args.num_classes, args.layers, args.dropout).to(args.device)
        elif args.model == 'MLP':
            model = MLP(data.x.shape[1], args.hidden, args.num_classes, args.layers, args.dropout).to(args.device)


        pbar = tqdm(range(args.runs), unit='run')
        for _ in pbar:
            
            seed_everything(args.seed + _)
            # print(args.seed + _)
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            best_val_loss = float('inf')
            val_loss_history = []

            for epoch in range(0, args.epochs):
                loss = train(model, data, optimizer, args)

                evals = evaluate(model, data, args)
    

                if loss['val'] < best_val_loss:
                    best_val_loss = loss['val']
                    test_acc = evals['test']
                    test_per_class_acc = evals['test_per_class']
                    f1_macro_score=evals['f1_macro']
                    gmean_score = evals['gmean']
                    bacc_score = evals['bacc']


                '''
                The code snippet implements early stopping based on the trend of validation loss.
                  If enabled and after the halfway point of training epochs, 
                  it calculates the mean of the recent validation loss values. 
                  If the current validation loss exceeds this mean, 
                  the training loop is terminated early to prevent overfitting.
                '''
                val_loss_history.append(loss['val'])
                if args.early_stopping > 0 and epoch > args.epochs // 2:
                    tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                    if loss['val'] > tmp.mean().item():
                        break


            accs[args.model][_] = test_acc.item()
            accs_per_class[args.model][_] = test_per_class_acc
            f1s_macro[args.model][_]=f1_macro_score
            gmeans[args.model][_] = gmean_score 
            baccs[args.model][_] = bacc_score 


            #print(test_acc.item())

    return accs, accs_per_class,f1s_macro,gmeans, baccs


if __name__ == '__main__':

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading dataset: ",args.dataset)
  
    
    # load llm
    if args.llm=="True" and not args.load_response:
        
        print("Using LLM:",args.model_name)
      
        args.tokenizer = AutoTokenizer.from_pretrained(args.model_name, load_in_8bit=False,torch_dtype=torch.float16)
        args.tokenizer.pad_token_id=args.tokenizer.eos_token_id
        args.llm_model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=False,torch_dtype=torch.float16).bfloat16()
        args.llm_model.to(args.device)
        print("Loaded the LLM model:",args.model_name)
        
        

    if args.model_name=="meta-llama/Meta-Llama-3-8B-Instruct":
        args.model_name=""
    else:
        args.model_name=args.model_name.split("/")[-1]
    
    
    # Pretrain the confidence function
    dataset_dir=os.path.join("CF",f'{args.dataset}_CF')
    encoder_path = os.path.join(dataset_dir, 'MLP_encoder.pt')
    predictor_path = os.path.join(dataset_dir, 'MLP_predictor.pt')
    
    if not os.path.exists(encoder_path) or not os.path.exists(predictor_path):
     
        args.pretrain=True
        train_cf(args)
        args.pretrain=False
    
    
        
    data, split_edge, args = get_dataset(args.dataset, args)
    accs, accs_per_class,f1s_macro,gmeans, baccs = run(data, args)
    

    for key in accs:
        print(f"-------------{key}-----------------")
        print("overall accuracy: ", np.mean(accs[key]), "overall std:", np.std(accs[key]))
        print("f1_score:",np.mean(f1s_macro[key]),"+-",np.std(f1s_macro[key]))
        print("BACC: ", np.mean(baccs[key]), "overall std:", np.std(baccs[key]))
        print("GMean:",np.mean(gmeans[key]),"+-",np.std(gmeans[key]))

        tmp = np.mean(accs_per_class[key], axis = 0)
        min_perform = tmp[args.imb_class].mean()
        maj_perform = (tmp.sum() - tmp[args.imb_class].sum())/ (len(tmp) - len(args.imb_class))
        print("Maj - Min Class Performance:", maj_perform - min_perform)
        print("Min: ",min_perform )
        print("Maj: ",maj_perform )

        
     
         
      