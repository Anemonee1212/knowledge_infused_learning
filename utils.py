"""
Helper functions

@author All team members
@date 12/15
"""
import numpy as np
import pandas as pd
import re
import sentence_transformers as st
import time

from typing import Any, Dict, List, Optional, Tuple

device = "cuda"
default_sys_prompt = """
You are an assistant of a cosmetic company. Please generate a description for a cosmetic product based on the
information provided, including the product name, product type, and product features that you should emphasize.
"""
bias_detector_sys_prompt = """
You are a bias detector for cosmetic product advertisements. I will provide you with a feature of a cosmetic product and
some entities from a knowledge graph. Note that the knowledge graph is toxic in nature. Please detect the type of bias
in these entities, and rewrite the product description avoiding that bias. Please respond in the following format:

Type of bias in entities: [some bias type, e.g., gender, race]
Sample unbiased description: [some cosmetic description without the above bias]
"""
label_to_bias_type = np.array([
    "toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit",
    "male", "female", "homosexual_gay_or_lesbian", "christian", "jewish", "muslim", "black", "white",
    "psychiatric_or_mental_illness"
])
bias_type_to_label = {
    "toxicity": 0, "severe_toxicity": 1, "obscene": 2, "threat": 3, "insult": 4, "identity_attack": 5,
    "sexual_explicit": 6, "male": 7, "female": 8, "homosexual_gay_or_lesbian": 9, "christian": 10, "jewish": 11,
    "muslim": 12, "black": 13, "white": 14, "psychiatric_or_mental_illness": 15
}


def timer(func):
    """
    A decorator function to show the execution time of other functions.
    """
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        sec = t2 - t1
        if sec >= 60:
            print(f"`{func.__name__}` executed in {sec / 60:.1f}min")
        else:
            print(f"`{func.__name__}` executed in {sec:.1f}s")

        return result

    return wrap_func


def construct_prompt(
        prod_name: str, prod_type: str, prod_features: str, sys_prompt: Optional[str] = None,
        bias_type: Optional[str] = None, unbiased_example: Optional[str] = None
) -> str:
    """
    Constructs prompt that let Llama 2 generate descriptions/advertisements.

    :param prod_name: Product name
    :param prod_type: Product type (e.g., moisturizer)
    :param prod_features: Product features which can be biased or unbiased
    :param sys_prompt: If provided, use as general instructions for LLaMA; otherwise, default version is used
    :param bias_type: If provided, instruct LLaMA to avoid this type of bias
    :param unbiased_example: If provided, instruct LLaMA to follow this example
    :returns: the full prompt passed to LLaMA
    """
    if sys_prompt is None:
        sys_prompt = default_sys_prompt

    user_prompt = f"""
    Product name: {prod_name}
    Product type: {prod_type}
    Features: {prod_features}
    """

    if bias_type is not None or unbiased_example is not None:
        user_prompt += f"""
        Do not include any {bias_type} bias in the feature provided. Refer to this example:
        {unbiased_example}
        """

    prompt_template = f"""
    <s>[INST] <<SYS>>{sys_prompt}<</SYS>>

    {user_prompt} [/INST]
    """
    return prompt_template


def construct_prompt_list(df: pd.DataFrame, include_bias: bool = False, include_instruct: bool = False) -> List[str]:
    """
    Construct a list of prompts passed to LLaMA

    :param df: DataFrame of product information
    :param include_bias: Indicating whether to include biased features
    :param include_instruct: Indicating whether to add KIL instructions
    :return:
    """
    prompt_list = []
    if include_instruct:  # With biased features, with KIL instruction
        for idx, row in df.iterrows():
            prompt = construct_prompt(
                prod_name = row["prod_name"], prod_type = row["prod_type"],
                prod_features = f"{row.unbiased_feature}; {row.biased_feature}",
                bias_type = row["bias_type"], unbiased_example = row["unbiased_example"]
            )
            prompt_list.append(prompt)

    elif include_bias:  # With biased features, without KIL instruction
        for idx, row in df.iterrows():
            prompt = construct_prompt(
                prod_name = row["prod_name"], prod_type = row["prod_type"],
                prod_features = f"{row.unbiased_feature}; {row.biased_feature}"
            )
            prompt_list.append(prompt)

    else:  # Without biased features, without KIL instruction
        for idx, row in df.iterrows():
            prompt = construct_prompt(
                prod_name = row["prod_name"], prod_type = row["prod_type"],
                prod_features = row["unbiased_feature"]
            )
            prompt_list.append(prompt)

    return prompt_list


def construct_prompt_instruction(biased_feature: str, entities: str) -> str:
    prompt_template = f"""<s>[INST] <<SYS>>{bias_detector_sys_prompt}<</SYS>>
    
    Cosmetic product feature: {biased_feature}
    Toxic entities: {entities} [/INST]
    """
    return prompt_template


def construct_prompt_list_instruction(
        data_feature: pd.DataFrame, data_entity: pd.DataFrame,
        sentence_encoder: st.SentenceTransformer
) -> list[str]:
    feature_entity_pairs = match_feature_entity(data_feature, data_entity, sentence_encoder)
    prompt_list = [construct_prompt_instruction(*feature_entity) for feature_entity in feature_entity_pairs]
    return prompt_list


def extract_response(sequences: List[Dict[str, Any]]) -> List[str]:
    responses = []
    for seq in sequences:
        text = seq["generated_text"]
        idx_response = text.find("[/INST]") + 8
        responses.append(text[idx_response:])

    return responses


def extract_instruction(response) -> Tuple[str, str, bool]:
    text = response[0]["generated_text"]
    idx_response = text.find("[/INST]") + 8
    instruction = text[idx_response:]
    bias_type_match = re.search(r'Type of bias in entities: (.+?)\n', instruction)
    if bias_type_match:
        bias_type = bias_type_match.group(1)
    else:
        bias_type = None
        print("No bias type match found.")

    unbiased_example_match = re.search(r'Sample unbiased description:\s+(.+?)(?:\n|$)', instruction)
    if unbiased_example_match:
        unbiased_example = unbiased_example_match.group(1)
    else:
        unbiased_example = None
        print("No unbiased example match found.")

    return bias_type, unbiased_example, bias_type is not None and unbiased_example is not None


def read_str_list_format(input_str: str) -> List[str]:
    pattern = r"(?<!\\)\"(.+?)(?<!\\)\"|(?<!\\)'(.+?)(?<!\\)'"
    elements = re.findall(pattern, input_str)

    str_list = []
    for ele1, ele2 in elements:
        if len(ele1) > 0:
            str_list.append(ele1)
        elif len(ele2) > 0:
            str_list.append(ele2)

    return str_list


def pivot_text_data(response: pd.Series) -> pd.DataFrame:
    data_score = pd.DataFrame(columns = ["prod_id", "text"])
    for idx, raw_text in enumerate(response):
        text_list = read_str_list_format(raw_text)
        new_data = pd.DataFrame({"prod_id": [idx] * len(text_list), "text": text_list})
        data_score = pd.concat([data_score, new_data], ignore_index = True)

    return data_score


def split_feature_list(df: pd.DataFrame) -> pd.Series:
    data_unbiased = df["unbiased_feature"].str.split("; ")
    data_biased = df["biased_feature"].str.split("; ")
    data_feature = pd.concat([data_biased, data_unbiased], axis = 1).apply(
        lambda row: row["biased_feature"] + row["unbiased_feature"], axis = 1
    )
    return data_feature


def find_nearest_neighbor(ft_emb: np.ndarray, ent_emb: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
    dist = np.linalg.norm(
        np.expand_dims(ft_emb, axis = 0) - np.expand_dims(ent_emb, axis = 1),
        axis = 2, ord = 2
    )
    nn_idx = np.unravel_index(np.argmin(dist), dist.shape)
    return nn_idx, dist


def find_entity(df: pd.DataFrame, entity: str) -> List[str]:
    contains = np.any([column == entity for _, column in df.items()], axis = 0)
    return df[contains].iloc[0].tolist()


def match_feature_entity(
        data_feature: pd.DataFrame, data_entity: pd.DataFrame,
        sentence_encoder: st.SentenceTransformer
) -> List[Tuple[str, List[str]]]:
    prod_features = split_feature_list(data_feature)

    entities = pd.concat([data_entity["Subject"], data_entity["Object"]], axis = 0).unique()
    entity_embeddings = sentence_encoder.encode(entities)

    feature_entity_pairs = []
    for idx, feature_list in enumerate(prod_features):
        feature_embeddings = sentence_encoder.encode(feature_list)
        nearest_neighbor_idx, distances = find_nearest_neighbor(feature_embeddings, entity_embeddings)
        biased_feature_detect = feature_list[nearest_neighbor_idx[1]]
        biased_entities_detect = find_entity(data_entity, entity = entities[nearest_neighbor_idx[0]])
        feature_entity_pairs.append((biased_feature_detect, biased_entities_detect))

    return feature_entity_pairs
