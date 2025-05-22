import argparse


parser = argparse.ArgumentParser()



# Setup  
parser.add_argument('--dataset', type=str, default='Cora',
                    help='Name of the dataset to use (Cora, Citeseer, PubMed, Photo, Computer, Child)')
parser.add_argument('--seed', type=int, default=1033,
                    help='Random seed for reproducibility')
parser.add_argument('--imb_ratio', type=float, default=0.2,
                    help='Imbalance ratio for minority classes')

# Interpolation
parser.add_argument('--k', type=int, default=3,
                    help='Number of top neighbors to consider for interpolation')
parser.add_argument('--method', type=str, default="none",
                    help='Interpolation method to use (e.g., "S", "M","O", "none")')
parser.add_argument('--llm', type=str, default="False",
                    help='Whether to use LLM-based augmentation (True/False)')
parser.add_argument('--edge', type=str, default="False",
                    help='Whether to generate edges for new nodes (True/False)')
parser.add_argument('--load_response', action='store_true',
                    help='Flag to load previously generated responses instead of regenerating')
parser.add_argument('--embed', type=str, default="SBERT",
                    help='Embedding type to use for text attributes (e.g., SBERT, TF-IDF)')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help='Name or path of the LLM to use for text generation')
parser.add_argument('--save', action='store_true',
                    help='Flag to save generated data and model checkpoints')

# Edge assignment
parser.add_argument('--encoder_lr', type=float, default=0.001,
                    help='Learning rate for the edge encoder')
parser.add_argument('--predictor_lr', type=float, default=0.001,
                    help='Learning rate for the link predictor')
parser.add_argument('--n_hidden', type=int, default=256,
                    help='Number of hidden units in the encoder')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of layers in the encoder network')
parser.add_argument('--en_dp', type=float, default=0.0,
                    help='Dropout rate for the encoder')
parser.add_argument('--lp_dp', type=float, default=0.0,
                    help='Dropout rate for the link predictor')
parser.add_argument('--train_bsz', type=int, default=1152,
                    help='Batch size for edge assignment training')
parser.add_argument('--eval_bsz', type=int, default=8192,
                    help='Batch size for edge assignment evaluation')
parser.add_argument('--track_idx', type=int, default=4,
                    help='Index of the feature to track or visualize')
parser.add_argument("--pretrain", action="store_true", default=False,
                    help="Pretraining the confidence function")

# For node classification
parser.add_argument('--runs', type=int, default=5,
                    help='Number of independent runs for evaluation')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate for node classification model')
parser.add_argument('--early_stopping', type=int, default=100,
                    help='Patience for early stopping (in epochs)')
parser.add_argument('--layers', type=int, default=2,
                    help='Number of layers in the classification model')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units in each layer of the model')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate for the classification model')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs')


args = parser.parse_args()