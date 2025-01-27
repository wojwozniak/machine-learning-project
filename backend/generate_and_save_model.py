import pickle
import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

# Load and prepare data
file_path = "./data"
anime = pd.read_csv(file_path + "/anime.csv")
ratings = pd.read_csv(file_path + "/rating.csv")

# Clean data
anime = anime.dropna(subset=["genre", "rating"])
ratings = ratings.dropna(subset=["anime_id", "user_id"])
anime = anime.dropna(subset=["anime_id"])
ratings = ratings[ratings["rating"] != -1]

# Ensure we only include anime_ids that exist in the anime dataset
valid_anime_ids = set(anime["anime_id"].unique())
ratings = ratings[ratings["anime_id"].isin(valid_anime_ids)]

# Create mappings
anime_id_to_index = {anime_id: idx for idx,
                     anime_id in enumerate(sorted(valid_anime_ids))}
user_id_to_index = {user_id: idx for idx,
                    user_id in enumerate(sorted(ratings["user_id"].unique()))}

# Add index mappings to ratings
ratings["anime_idx"] = ratings["anime_id"].map(anime_id_to_index)
ratings["user_idx"] = ratings["user_id"].map(user_id_to_index)
ratings = ratings.dropna(subset=["anime_idx", "user_idx"])

# Create sparse matrix of user-anime interactions
interaction_matrix = coo_matrix(
    (ratings["rating"], (ratings["user_idx"], ratings["anime_idx"])),
    shape=(len(user_id_to_index), len(anime_id_to_index)),
).tocsr()

# Apply confidence weighting
interaction_matrix.data = 1 + np.log1p(interaction_matrix.data)

# Limit OpenBLAS threads for performance
os.environ["OPENBLAS_NUM_THREADS"] = "1"
with threadpool_limits(limits=1, user_api="blas"):
    model = AlternatingLeastSquares(
        factors=50, regularization=0.01, iterations=15)
    model.fit(interaction_matrix)

# Save the model to disk using pickle
with open("als_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Also save the mappings to disk
with open("anime_id_to_index.pkl", "wb") as f:
    pickle.dump(anime_id_to_index, f)

with open("user_id_to_index.pkl", "wb") as f:
    pickle.dump(user_id_to_index, f)

print("Model training and saving completed!")
