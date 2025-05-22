# SaVe-TAG

This GitHub repo is the code implementation of the paper "SaVe-TAG: 
Semantic-aware Vicinal Risk Minimization for Long-Tailed Text-Attributed Graphs".

### Environment Configuration
Install the following environment:
```
# Install using conda ( CUDA 12.1 )
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# Install using pip ( CUDA 12.1 )
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch_geometric
pip install sentence_transformers
pip install "numpy<2.0"
pip install imblearn
```
### Download datasets
You can access our preprocessed dataset here: [SaVe-TAG-processed dataset](https://www.dropbox.com/scl/fi/nktqtna8httsvvkehp2x2/dataset.zip?rlkey=4vipqaa6bdtqkkzfvk5r4gli1&st=ue5dmc1z&dl=0). Then added downloaded `.pt` file under the folder "dataset".

### Run Experiments 
We have included the pretrained confidence function in the directory so that experiments can be conducted directly using the following command:
```
./run.sh
```

### Pretrain Link Predictor
We have uploaded pretrained confidence functions for each dataset in the `./CF` directory. To train a new link predictor, simply delete the corresponding `.pt` files and run `./run.sh.` The code will automatically pretrain new confidence functions if they are not found.

