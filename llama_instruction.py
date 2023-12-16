"""
Inspired by Anne's idea, use SentenceBERT to find the closest matching of (biased_feature, toxic_entity) pair, and let
LLaMA to detect its bias type and generate a response without bias.

@author Anne Wei, Colin Zhang, Robert Shi
@date 12/15/2023
"""
import torch
import transformers

from utils import *

# ========== Hyperparameters ==========
max_len = 500
model_name_llama = "meta-llama/Llama-2-7b-chat-hf"
model_name_sentence_encoder = "all-MiniLM-L6-v2"
num_text_generated = 5
temperature = 0.9


# ========== Helper Function ==========
@timer
def llama_pipeline_instruction(
        model: transformers.pipelines, tokenizer: Any, prompt: List[str]
) -> Tuple[List[str], List[str], List[bool]]:
    instruction_list = model(
        prompt, do_sample = True, temperature = temperature, top_k = 50, top_p = 0.9,
        num_return_sequences = 1, eos_token_id = tokenizer.eos_token_id, max_length = max_len
    )
    # response_list is a list of tuples (bias_type, unbiased_example, detected)
    response_list = [extract_instruction(instr) for instr in instruction_list]
    return tuple(zip(*response_list))


def generate_valid_instructions(model: transformers.pipelines, tokenizer: Any, prompt: List[str]) -> pd.DataFrame:
    """
    Calling `llama_pipeline_instruction` repeatedly until we extract all valid instructions. It is necessary because we
    cannot guarantee that LLaMA always generates instructions as expected.
    """
    data_gen = pd.DataFrame({"prompt": prompt, "bias_type": None, "unbiased_example": None, "match": False})
    while not np.all(data_gen["match"]):  # While empty elements still exist
        prompt_unmatch = data_gen[~data_gen["match"]]["prompt"].tolist()  # Get rows with empty elements
        bias_type, unbiased_example, match = llama_pipeline_instruction(model, tokenizer, prompt_unmatch)
        print(f"Successfully generated {sum(match)}/{len(prompt_unmatch)} instructions")

        # Write results
        data_gen.loc[~data_gen["match"], "bias_type"] = bias_type
        data_gen.loc[~data_gen["match"], "unbiased_example"] = unbiased_example
        data_gen.loc[~data_gen["match"], "match"] = match

    return data_gen[["bias_type", "unbiased_example"]]


if __name__ == "__main__":
    # Read data
    data = pd.read_csv("data/prod_data.csv")
    triplets = pd.read_csv("data/kg_triplets.csv")

    # Load SentenceBERT model
    sent_tf = st.SentenceTransformer(model_name_sentence_encoder)

    # Use SentenceBERT model to form the prompt
    prompt_list = construct_prompt_list_instruction(
        data_feature = data, data_entity = triplets, sentence_encoder = sent_tf
    )

    # Load LLaMA model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_llama)
    llama_pipeline = transformers.pipeline(
        "text-generation",
        model = model_name_llama,
        torch_dtype = torch.float16,
        device_map = "auto",
    )

    # Use LLaMA to generate instructions
    print("Session Initiated.")
    response_list = generate_valid_instructions(model = llama_pipeline, tokenizer = tokenizer, prompt = prompt_list)

    # Save data
    data_inst = pd.concat([data, response_list], axis = 1)
    data_inst.to_csv("data/prod_data_kil_instruction.csv", index = False)
    print("Session Terminated.")
