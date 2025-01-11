import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits
import os

# pip install implicit pandas numpy scipy

# get data
file_path = "../data"
anime = pd.read_csv(file_path + "/anime.csv")
ratings = pd.read_csv(file_path + "/rating.csv")

# drop rows with missing values in 'genre' or 'rating' and missing ids
anime = anime.dropna(subset=["genre", "rating"])
ratings = ratings.dropna(subset=["anime_id", "user_id"])
anime = anime.dropna(subset=["anime_id"])

# drop watched anime without rating
ratings = ratings[ratings["rating"] != -1]

# create a mapping of anime_id to index for matrix creation
anime_id_to_index = {
    anime_id: idx for idx, anime_id in enumerate(anime["anime_id"].unique())
}
user_id_to_index = {
    user_id: idx for idx, user_id in enumerate(ratings["user_id"].unique())
}

# add index mappings to the ratings dataframe
ratings["anime_idx"] = ratings["anime_id"].map(anime_id_to_index)
ratings["user_idx"] = ratings["user_id"].map(user_id_to_index)

# drop missing ids again
ratings = ratings.dropna(subset=["anime_idx", "user_idx"])

"""
# check for invalid mappings
print(ratings['anime_id'].isin(anime_id_to_index.keys()).all())  # should return True
print(ratings['user_id'].isin(user_id_to_index.keys()).all())   # should return True
"""

# create a sparse matrix of user-anime interactions
interaction_matrix = coo_matrix(
    (ratings["rating"], (ratings["user_idx"], ratings["anime_idx"])),
    shape=(len(user_id_to_index), len(anime_id_to_index)),
)

# print("Interaction Matrix Shape:", interaction_matrix.shape)

# TRAIN MODEL

# limit OpenBLAS threads for performance
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# convert to implicit format (confidence matrix)
# adding a confidence weight for the interaction (logarithmic scale)

interaction_matrix_conf = (
    interaction_matrix.copy()
)  # start with the original sparse matrix

# Convert to binary (implicit feedback)
interaction_matrix_conf.data = np.ones_like(interaction_matrix.data)

# Apply confidence weighting
interaction_matrix_conf.data = 1 + np.log(1 + interaction_matrix.data)

# Convert interaction matrix to CSR format
interaction_matrix_conf_csr = interaction_matrix_conf.tocsr()

# Initialize the ALS model
als_model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15)

# Train the model with proper threading limits
with threadpool_limits(limits=1, user_api="blas"):
    als_model.fit(interaction_matrix_conf_csr.T)

print("ALS model training completed!")

