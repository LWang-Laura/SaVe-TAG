import torch
import time




def form_prompt(text0,text1,category0, category1,args):
   
    prompt=form_llama_prompt(text0,text1,category0, category1,args)
    return prompt
    
    
def form_llama_prompt(text0,text1,category0, category1,args):

    if args.dataset =="Cora" or args.dataset =="Citeseer":
        text_format= "[New Title] : [New Abstract]\n"
        text="article"
        task="new academic articles"
        
    elif args.dataset =="PubMed":
        text_format= "Title: [New Title]\nAbstract: [New Abstract]"
        text="article"
        task="new academic articles"
        
    elif args.dataset=="Child" :
        text_format= "Title: [New Title]\nBook Description: [New Description]"
        text="book description"
        task="new book descriptions"
        
    elif args.dataset in ["Photo", "Computer"]:
        text_format="Review: [New Review]"
        text="review"
        task="reviews of products from Amazon"
    
        
    
    dataset_name=args.dataset
    
    if args.method=="zero_shot":
        prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for generating {task} from {dataset_name}, where each {text} are in the format '<START>{text_format}<END>'.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Give me one {text} from {dataset_name} with the topic '{category0}'.<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
'''

    elif args.method=="few_shots":
        prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for generating {task} from {dataset_name}, where each {text} are in the format '<START>{text_format}<END>'.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Give me one {text} from {dataset_name} with the topic '{category0}'.<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>

<START>{text0}<END>
<|start_header_id|>user<|end_header_id|>

Give me one {text} from {dataset_name} with the topic '{category0}'.<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>

<START>{text1}<END>
<|start_header_id|>user<|end_header_id|>

Give me one {text} from {dataset_name} with the topic '{category0}'.<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
'''
        
        
    elif args.method=="O":
        
        prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for generating {task} from {dataset_name}, where each {text} are in the format '<START>{text_format}<END>'.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Give me the first {text} from {dataset_name} with the topic '{category0}'.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

<START>{text0}<END>
<|start_header_id|>user<|end_header_id|>

Give me the second {text} from {dataset_name} with the topic '{category0}'. It should be more similar to the first {text}.<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
'''
        
        
    
    else:
        prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for generating {task} from {dataset_name}, where each {text} are in the format '<START>{text_format}<END>'.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Give me the first {text} from {dataset_name} with the topic '{category0}'.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

<START>{text0}<END>
<|start_header_id|>user<|end_header_id|>

Give me the second {text} from {dataset_name} with the topic '{category1}'.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

<START>{text1}<END>
<|start_header_id|>user<|end_header_id|>

Give me the third {text} from {dataset_name} with the topic '{category0}'. It should be more similar to the first {text} and less similar to the second {text}.<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
'''
   
    return prompt



def process_prompt_list(prompt_list,args,batch_size):
    answer_list=[]
    output_list=[]
    total_time=0
    cnt=0
   
    i=0
    while i<(len(prompt_list)):
        if i+batch_size >len(prompt_list):
            batch_data=prompt_list[i:]
        else:
            batch_data=prompt_list[i:i+batch_size]
        
        start_time = time.time()
        answers , outputs=get_batched_response(batch_data,output_list,args,batch_size)
        end_time = time.time()
        execution_time = end_time - start_time
        print("LLM Inference Time: ",execution_time )
        total_time+= execution_time 
        cnt+=1
        
        answer_list.extend(answers)
        output_list.extend(outputs)
        i+=batch_size
        
    print(f"LLM Average Inferece time with a batch size of {batch_size} is {total_time/cnt}")
    
    return answer_list, output_list





def get_batched_response(batch_data,output_list,args,batch_size):
    answer_list=[]
    output_list=[]
    
    args.tokenizer.pad_token = args.tokenizer.eos_token  # Most LLMs don't have a pad token by default
    model_inputs = args.tokenizer(batch_data, return_tensors="pt", padding=True).to(args.device)
    generated_ids = args.llm_model.generate(**model_inputs, max_new_tokens=300)
    generated_text_list=args.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    for generated_text in generated_text_list:
        answer=generated_text.split("<START>")[-1]
        answer=answer.split("<END>")[0].strip(" ")
        answer_list.append(answer)
        # print(generated_text)
        output={"generated_text":generated_text,"answer":answer}
        output_list.append(output)
        
    return answer_list,output_list