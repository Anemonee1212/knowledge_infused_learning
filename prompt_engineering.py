"""prompt_tuning.py
Author: Colin Zhang
Last Modified: 11/23/2023
Description: This script is used as a pipeline to tune the prompt 
(zero- vs one- vs few-shot learning) for Llama model. Colin wrote
this code by mimicking the code from llama_no_kil_response.py.
"""

# ========== Import packages ==========
import transformers
import torch

from utils import *

# ========== Hyperparameters ==========
max_len = 500
num_text_generated = 5
temperature = 0.9

# ========== Helper Function ==========
@timer
def llama_response(model: transformers.pipelines, tokenizer: Any, prompt: List[str]) -> List[List[str]]:
    sequence_list = model(
        prompt, temperature = temperature, do_sample = True, top_k = 50, top_p = 0.9,
        num_return_sequences = num_text_generated, eos_token_id = tokenizer.eos_token_id, max_length = max_len
    )
    response_list = [extract_response(sequences) for sequences in sequence_list]
    return response_list

def construct_prompt_examples(sys_prompt: str, user_prompt: str, example: None or str = None) -> str:
    # update sys_prompt if example is not None -> one-shot or few-shot learning
    if example is not None:
        sys_prompt = f"{sys_prompt} \nHere are some examples:: \n{example}"
        
    prompt_template = f"""
    <s>[INST] <<SYS>>{sys_prompt}<</SYS>>

    {user_prompt} [/INST]
    """
    return prompt_template

def construct_prompt_examples_list(df: pd.DataFrame, sys_prompt: str, example: None or str = None) -> List[str]:
    prompt_list = []
    for idx, row in df.iterrows():
        prompt = construct_prompt_examples(
            sys_prompt = sys_prompt, user_prompt = row["text"], example = example
        )
        prompt_list.append(prompt)
    return prompt_list

"""
Main method
"""
if __name__ == "__main__":
    # 1. Read and preprocess data
    data = pd.read_csv("data/response_score_data_biased.csv")
    
    # 2. construct prompt list
    # !!! WORK ON THIS !!!
    # NEED to use KG to generate one-shot and few-shot examples in the futures.
    sys_prompt = '''You are an assistant of a cosmetic company. You will be given a description for a cosmetric product.
    The given description may or may not contain some bias including gender, race, and skin etc. you job is to rewrite the description
    and remove the bias while maintaining the meaning of the description.'''
    
    one_shot_example = '''
    Introducing the In-shower style fixer, a revolutionary hair styling product that will elevate your feminine charm with its powerful blend of natural ingredients.
    ==>
    Introducing the In-shower style fixer, a revolutionary hair styling product that will elevate your charm with its powerful blend of natural ingredients.'''
    
    few_shot_example ='''
    Introducing the In-shower style fixer, a revolutionary hair styling product that will elevate your feminine charm with its powerful blend of natural ingredients.
    ==>
    Introducing the In-shower style fixer, a revolutionary hair styling product that will elevate your charm with its powerful blend of natural ingredients.

    With a fresh, clean scent that's designed to appeal to men, this deodorant is the perfect choice for anyone who wants to smell like a man who knows how to smell manly.
    ==>
    With a fresh, clean scent that's designed to appeal to men, this deodorant is the perfect choice for anyone who prefer a robust and traditionally masculine scent.

    This revolutionary deodorant offers a range of features that set it apart from other products on the market, making it the ideal choice for women with skin sensitivity concerns.
    ==>
    This revolutionary deodorant offers a range of features that set it apart from other products on the market, making it the ideal choice for women to improve its sensitivity.
    '''
    
    zero_shot_prompt_list = construct_prompt_examples_list(data, sys_prompt = sys_prompt, example = None)
    one_shot_prompt_list = construct_prompt_examples_list(data, sys_prompt = sys_prompt, example = one_shot_example)
    few_shot_prompt_list = construct_prompt_examples_list(data, sys_prompt = sys_prompt, example = few_shot_example)
    
    # 3. Load LLaMA model
    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    llama_pipeline = transformers.pipeline(
        "text-generation", model = model,
        torch_dtype = torch.float16, device_map = "auto"
    )
    
    # 4. Generate response
    print("Session Initiated.")
    print(">>> Zero-shot:")
    zero_shot_response = llama_response(model = llama_pipeline, tokenizer = tokenizer, prompt = zero_shot_prompt_list)
    print(">>> One-shot:")
    one_shot_response = llama_response(model = llama_pipeline, tokenizer = tokenizer, prompt = one_shot_prompt_list)
    print(">>> Few-shot:")
    few_shot_response = llama_response(model = llama_pipeline, tokenizer = tokenizer, prompt = few_shot_prompt_list)
    
    # 5. Save data
    data["zero_shot_response"] = zero_shot_response
    data["one_shot_response"] = one_shot_response
    data["few_shot_response"] = few_shot_response
    data.to_csv("data/response_score——data_prompt_engineering.csv", index = False)
    print("Session Terminated.")