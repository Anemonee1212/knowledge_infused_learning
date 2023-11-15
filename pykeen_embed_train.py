import pandas as pd
import pickle

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Hyperparameters
device = "cuda"
embed_dim = 32
n_epochs = 300

# Load Data
triplets = pd.read_csv("data/triplets.csv")
dataset = TriplesFactory.from_labeled_triples(triplets.values)

# Training Session
results = pipeline(
    model = "TransE", training = dataset, testing = dataset,
    model_kwargs = dict(embedding_dim = embed_dim),
    training_kwargs = dict(num_epochs = n_epochs, use_tqdm_batch = True),
    random_seed = 3407, device = device
)

# Save embeddings
embeddings = results.model.entity_representations[0](indices = None).cpu().detach()
embed_dict = {}
for entity, idx in dataset.entity_to_id.items():
    embed_dict[entity] = embeddings[idx, :]

with open("data/pykeen_embed.pkl", "wb") as handle:
    pickle.dump(embed_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
