# Bias Evaluation
import torch
import torch.nn.functional as F
import transformers

from utils import *

# ========== Hyperparameters ==========
response_type = "biased"


# ========== Helper Function ==========
@timer
def predict_bias_score(model: transformers.models, tokenizer: Any, text: List[str]) -> torch.tensor:
    tokenized_text = tokenizer(text, return_tensors = "pt", padding = True, truncation = True).to(device)
    pred_bias = model(**tokenized_text)
    bias_score = F.sigmoid(pred_bias.logits).max(dim = 1)
    return bias_score


"""
Main method
"""
if __name__ == "__main__":
    # Read and preprocess data
    data = pd.read_csv("data/response_data.csv")
    data_response = pivot_text_data(data[f"{response_type}_response"])

    # Load RoBERTa model
    model = "unitary/unbiased-toxic-roberta"
    bias_model = transformers.AutoModelForSequenceClassification.from_pretrained(model).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    # Predict bias score
    bias_score = predict_bias_score(model = bias_model, tokenizer = tokenizer, text = data_response.text.tolist())

    # Save data
    data_response["score"] = bias_score.values.cpu().detach()
    data_response["type"] = label_to_bias_type[bias_score.indices.cpu().tolist()]
    data_response.to_csv(f"data/response_score_data_{response_type}.csv", index = False)
    print("Session Terminated.")
