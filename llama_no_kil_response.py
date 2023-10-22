# Sample code
import pandas as pd
import time
import transformers
import torch

from typing import List

# ========== Hyperparameters ==========
max_len = 500
num_text_generated = 3


# ========== Helper functions ==========
def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"`{func.__name__}` executed in {(t2 - t1):.1f}s")
        return result

    return wrap_func


def construct_prompt(
        sys_prompt: str, prod_name: str, prod_type: str,
        unbiased_feature: str, biased_feature: str
) -> str:
    user_prompt = f"""
    Product name: {prod_name}
    Product type: {prod_type}
    Features: {unbiased_feature + "; " + biased_feature}
    """
    prompt_template = f"""
    <s>[INST] <<SYS>>{sys_prompt}<</SYS>>

    {user_prompt} [/INST]
    """
    return prompt_template


@timer
def llama_response(prompt: str) -> List[str]:
    sequences = llama_pipeline(
        prompt, do_sample = True, top_k = 10, num_return_sequences = num_text_generated,
        eos_token_id = tokenizer.eos_token_id, max_length = max_len
    )

    responses = []
    for seq in sequences:
        text = seq["generated_text"]
        idx_response = text.find("[/INST]") + 8
        responses.append(text[idx_response:])

    return responses


"""
Main method
"""
if __name__ == "__main__":
    # Read data
    data = pd.read_excel("data/input_data.xlsx")

    # Load LLaMA model
    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    llama_pipeline = transformers.pipeline(
        "text-generation", model = model,
        torch_dtype = torch.float16, device_map = "auto"
    )

    sys_prompt = """
    You are an assistant of a cosmetic company. You will be provided a description of a cosmetic product, including the
    product name, product type and product features. Then you will write an attractive advertisement for that product.
    The advertisement you wrote should be complete, attractive, and highlight the product’s features as much as
    possible.
    """

    # Traverse through dataset
    print("Session Initiated.")
    data["responses"] = ""
    for idx, row in data.iterrows():
        prompt = construct_prompt(
            sys_prompt, prod_name = row["prod_name"], prod_type = row["prod_type"],
            unbiased_feature = row["unbiased_feature"], biased_feature = row["biased_feature"]
        )
        responses = llama_response(prompt)
        data.at[idx, "responses"] = responses

    # Save data
    data.to_excel("data/output_data.xlsx")
    print("Session Terminated.")
