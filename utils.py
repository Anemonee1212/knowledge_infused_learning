# Helper functions
import pandas as pd
import time

from typing import Any, Dict, List

default_sys_prompt = """
You are an assistant of a cosmetic company. Please generate a description for a cosmetic product based on the
information provided, including the product name, product type, and product features that you should emphasize.
"""


def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"`{func.__name__}` executed in {(t2 - t1) / 60:.1f}min")
        return result

    return wrap_func


def construct_prompt(prod_name: str, prod_type: str, prod_features: str, sys_prompt: str = None) -> str:
    if sys_prompt is None:
        sys_prompt = default_sys_prompt

    user_prompt = f"""
    Product name: {prod_name}
    Product type: {prod_type}
    Features: {prod_features}
    """
    prompt_template = f"""
    <s>[INST] <<SYS>>{sys_prompt}<</SYS>>

    {user_prompt} [/INST]
    """
    return prompt_template


def construct_prompt_list(df: pd.DataFrame, include_bias: bool = False) -> List[str]:
    prompt_list = []
    if include_bias:
        for idx, row in df.iterrows():
            prompt = construct_prompt(
                prod_name = row["prod_name"], prod_type = row["prod_type"],
                prod_features = f"{row.unbiased_feature}; {row.biased_feature}"
            )
            prompt_list.append(prompt)

    else:
        for idx, row in df.iterrows():
            prompt = construct_prompt(
                prod_name = row["prod_name"], prod_type = row["prod_type"],
                prod_features = row["unbiased_feature"]
            )
            prompt_list.append(prompt)

    return prompt_list


def extract_response(sequences: List[Dict[str, Any]]) -> List[str]:
    responses = []
    for seq in sequences:
        text = seq["generated_text"]
        idx_response = text.find("[/INST]") + 8
        responses.append(text[idx_response:])

    return responses
