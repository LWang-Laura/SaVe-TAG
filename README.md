# SaVe-TAG

This GitHub repo is the code implementation of the paper "SaVe-TAG: 
Semantic-aware Vicinal Risk Minimization for Long-Tailed Text-Attributed Graphs".

### Install Environment
Install with `pip install -r requirements.txt` or manually install the following environment:
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
You can access our preprocessed dataset here: [SaVe-TAG-processed dataset](https://www.dropbox.com/scl/fi/nktqtna8httsvvkehp2x2/dataset.zip?rlkey=4vipqaa6bdtqkkzfvk5r4gli1&st=ue5dmc1z&dl=0). Then added downloaded `.pt` file under the folder `./dataset`.

If you want to download the raw data, please visit https://github.com/CurryTang/Graph-LLM and https://huggingface.co/datasets/zkchen/tsgfm/blob/main/minilmdata.zip. Our code for preprocessing has been uploaded in `dataset/preprocess.ipynb`.


### Run Experiments 
We have included the pretrained confidence function in the directory so that experiments can be conducted directly using the following command:
```
./run.sh
```

**Load generated response**
After running for the first time, the code will automatically saved in the folder `./llm_response`. You can just load the generated data to save time from regenerating the texts again. 

*Initial generation / Regenerating the texts with LLM*: 
```
python src/main.py --dataset  $name  --llm="True"  --method $mtd --CF="True" 
```

*Loading the generated texts*: 
```
python src/main.py --dataset  $name  --llm="True"  --method $mtd --CF="True" --load_response 
```

### Optional: Pretrain Confidence Function
We have uploaded pretrained confidence functions for each dataset in the `./CF` directory. To train a new confidence function, simply delete the corresponding `.pt` files and run `./run.sh.` The code will automatically pretrain confidence functions if they are not found.

