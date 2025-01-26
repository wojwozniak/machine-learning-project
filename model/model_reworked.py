import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits
import os

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

# Create reverse mappings
index_to_anime_id = {idx: anime_id for anime_id,
                     idx in anime_id_to_index.items()}
index_to_user_id = {idx: user_id for user_id, idx in user_id_to_index.items()}

# Add index mappings to ratings
ratings["anime_idx"] = ratings["anime_id"].map(anime_id_to_index)
ratings["user_idx"] = ratings["user_id"].map(user_id_to_index)
ratings = ratings.dropna(subset=["anime_idx", "user_idx"])

# Validate mappings
print("Validation checks:")
print("All anime IDs valid:", ratings['anime_id'].isin(
    anime_id_to_index.keys()).all())
print("All user IDs valid:", ratings['user_id'].isin(
    user_id_to_index.keys()).all())
print("Number of unique anime:", len(anime_id_to_index))
print("Number of unique users:", len(user_id_to_index))

# Create sparse matrix of user-anime interactions
interaction_matrix = coo_matrix(
    (ratings["rating"], (ratings["user_idx"], ratings["anime_idx"])),
    shape=(len(user_id_to_index), len(anime_id_to_index)),
).tocsr()

print("Interaction Matrix Shape:", interaction_matrix.shape)

# Apply confidence weighting
interaction_matrix.data = 1 + np.log1p(interaction_matrix.data)

# Limit OpenBLAS threads for performance
os.environ["OPENBLAS_NUM_THREADS"] = "1"
with threadpool_limits(limits=1, user_api="blas"):
    model = AlternatingLeastSquares(
        factors=50, regularization=0.01, iterations=15)
    # Note: we're fitting on the original matrix, not the transposed one
    model.fit(interaction_matrix)

print("ALS model training completed!")


def recommend_anime(user_id, model, interaction_matrix, anime_df, n_recommendations=10):
    """
    Generate anime recommendations for a specific user.

    Parameters:
    -----------
    user_id : int
        The ID of the user to generate recommendations for
    model : AlternatingLeastSquares
        Trained ALS model
    interaction_matrix : scipy.sparse.csr_matrix
        User-anime interaction matrix in CSR format
    anime_df : pandas.DataFrame
        DataFrame containing anime metadata
    n_recommendations : int
        Number of recommendations to generate

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing recommended anime with their metadata and scores
    """
    try:
        # Convert user_id to internal user index
        if user_id not in user_id_to_index:
            raise KeyError(f"User ID {user_id} not found in training data")

        user_idx = user_id_to_index[user_id]

        # Get user recommendations
        user_items = interaction_matrix[user_idx]

        # Get recommendations
        ids, scores = model.recommend(
            user_idx,
            user_items,
            N=n_recommendations,
            filter_already_liked_items=True,
            recalculate_user=True
        )

        # Map anime indices back to anime_ids
        recommended_anime_ids = [index_to_anime_id[idx] for idx in ids]

        # Get recommended anime details
        recommended_anime = anime_df[anime_df['anime_id'].isin(
            recommended_anime_ids)].copy()

        # Add recommendation scores
        score_dict = dict(zip(recommended_anime_ids, scores))
        recommended_anime['score'] = recommended_anime['anime_id'].map(
            score_dict)

        # Sort by score and select relevant columns
        result = recommended_anime[['name', 'genre', 'type', 'rating', 'score']].sort_values(
            by='score', ascending=False
        )

        return result

    except KeyError as e:
        print(f"Error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return pd.DataFrame()


# Test the recommendations
test_user_id = 1
recommendations = recommend_anime(
    test_user_id, model, interaction_matrix, anime)

if not recommendations.empty:
    print(f"\nRecommendations for User {test_user_id}:")
    print(recommendations)
else:
    print(f"\nNo recommendations available for User {test_user_id}")
