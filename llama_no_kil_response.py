"""
Baseline approach to use LLaMA 2 model to generate raw descriptions without Knowledge-Infused Learning.

@author Robert Shi
@date 12/04/2023
"""
import transformers
import torch

from utils import *

# ========== Hyperparameters ==========
max_len = 500
model_name = "meta-llama/Llama-2-7b-chat-hf"
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


"""
Main method
"""
if __name__ == "__main__":
    # Read and preprocess data
    data = pd.read_csv("data/prod_data.csv")
    biased_prompt_list = construct_prompt_list(data, include_bias = True)
    unbiased_prompt_list = construct_prompt_list(data, include_bias = False)

    # Load LLaMA model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llama_pipeline = transformers.pipeline(
        "text-generation", model = model_name,
        torch_dtype = torch.float16, device_map = "auto"
    )

    # Generate response
    print("Session Initiated.")
    print(">>> Biased:")
    biased_response = llama_response(model = llama_pipeline, tokenizer = tokenizer, prompt = biased_prompt_list)
    print(">>> Unbiased:")
    unbiased_response = llama_response(model = llama_pipeline, tokenizer = tokenizer, prompt = unbiased_prompt_list)

    # Save data
    data["biased_response"] = biased_response
    data["unbiased_response"] = unbiased_response
    data.to_csv("data/response_data_no_kil.csv", index = False)
    print("Session Terminated.")
