# Llama
# python src/main.py --dataset $name --llm True --method $mtd --CF True \
#   --llm_model llama \
#   --llm_model_path meta-llama/Meta-Llama-3-8B-Instruct

# Qwen (Qwen2.5 Instruct)
# python src/main.py --dataset $name --llm True --method $mtd --CF True \
#   --llm_model qwen \
#   --llm_model_path Qwen/Qwen2.5-7B-Instruct \





LOG_FILE="logs/qwen.log"
exec > >(tee ${LOG_FILE}) 2>&1

for name in  Cora # PubMed Citeseer # Photo Computer Child  
do




        for mtd   in S  #O   M     
        do   

        
                echo "=== $name _with_confidence-function==="
        


                echo '---''SaVe-TAG (LLM_C)' $mtd '---'
                #python src/main.py --dataset  $name  --llm="True"  --method $mtd --CF="True" #--load_response  
                # Qwen (Qwen2.5 Instruct)
                #     python src/main.py --dataset $name --llm True --method $mtd --CF True \
                #     --llm_model qwen \
                #     --llm_model_path  Qwen/Qwen2.5-7B-Instruct #Qwen/Qwen3-30B-A3B-Instruct-2507 #Qwen/Qwen2.5-7B-Instruct \

                python src/main.py --dataset $name --llm True --method $mtd --CF True \
                  --llm_model llama \
                  --llm_model_path meta-llama/Meta-Llama-3-8B-Instruct

        done
done
