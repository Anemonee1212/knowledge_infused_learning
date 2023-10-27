# Sample code
import transformers
import torch

from utils import *

# ========== Hyperparameters ==========c
max_len = 500
num_text_generated = 5
temperature = 0.9


@timer
def llama_response(model: transformers.pipelines, prompt: List[str]) -> List[List[str]]:
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
    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    llama_pipeline = transformers.pipeline(
        "text-generation", model = model,
        torch_dtype = torch.float16, device_map = "auto"
    )

    # Generate response
    print("Session Initiated.")
    print(">>> Biased:")
    biased_response = llama_response(model = llama_pipeline, prompt = biased_prompt_list)
    print(">>> Unbiased:")
    unbiased_response = llama_response(model = llama_pipeline, prompt = unbiased_prompt_list)

    # Save data
    data["biased_response"] = biased_response
    data["unbiased_response"] = unbiased_response
    data.to_csv("data/response_data.csv", index = False)
    print("Session Terminated.")
