
for name in  Cora PubMed Citeseer  Photo Computer Child  
do
        # echo "===' $name 'without_confidence-function==="


        # echo '---Origin ---'
        #     python src/main.py --dataset  $name 



        for mtd   in O  S  M     
        do   

                # echo '---'NUM $mtd '---'
                # python src/main.py --dataset  $name --method $mtd  

                # echo '---''LLM' $mtd '---'
                # python src/main.py  --dataset  $name  --llm="True"  --method $mtd #--load_response # --save

             
        
                echo "=== $name _with_confidence-function==="
        

                # echo '---'NUM_C $mtd '---'
                # python src/main.py --dataset  $name --method $mtd  --CF="True"

                echo '---''SaVe-TAG (LLM_C)' $mtd '---'
                python src/main.py --dataset  $name  --llm="True"  --method $mtd --CF="True" #--load_response  


        done
done


