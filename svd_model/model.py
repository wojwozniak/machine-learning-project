import pandas as pd
import numpy as np
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import Reader, Dataset

# pip install implicit pandas numpy<2.0 scipy 
# pip install svd

ratings = "../data/anime.csv"
anime = "../data/rating.csv"

# remove unrated anime
df_ratings = pd.read_csv(ratings, sep=",", names=["user_id" ,"anime_id" ,"rating"], skiprows=1)
df_anime = pd.read_csv(anime, sep=",", names=["anime_id","name", "genre","type","episodes","rating","members"], skiprows=1)

# clean data
df_anime = df_anime.dropna(subset=["genre", "rating"])
df_ratings = df_ratings.dropna(subset=["anime_id", "user_id"])
df_anime = df_anime.dropna(subset=["anime_id"])
watched_anime = df_ratings
df_ratings = df_ratings[df_ratings["rating"] != -1]

# Ensure we only include anime_ids that exist in the anime dataset
valid_anime_ids = set(df_anime["anime_id"].unique())
df_ratings = df_ratings[df_ratings["anime_id"].isin(valid_anime_ids)]
watched_anime = watched_anime[watched_anime["anime_id"].isin(valid_anime_ids)]


reader = Reader(rating_scale=(1,10))
data = Dataset.load_from_df(df_ratings, reader).build_full_trainset()

#Train model
svd = SVD(n_factors=100, reg_all=0.05)
svd.fit(data)

print("SVD training completed!")

def recommend_anime(user_id, model, ratings_df, anime_df, n_recommendations=10):
    """
    Generate anime recommendations for a specific user.

    Parameters:
    -----------
    user_id : int
        The ID of the user to generate recommendations for
    model : SVD
        Trained singular value decomposition model
    ratings_df : pandas.DataFrame
        DataFrame containing ratings metadata
    anime_df : pandas.DataFrame
        DataFrame containing anime metadata
    n_recommendations : int
        Number of recommendations to generate

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing recommended anime with their metadata and scores
    """
    # Get all unique anime IDs
    all_anime_ids = anime_df["anime_id"].unique()

    # Get the anime already watched by the user
    rated_anime_ids = watched_anime[watched_anime["user_id"] == user_id]["anime_id"].unique()

    # Filter out anime already watched by the user
    unrated_anime_ids = [anime_id for anime_id in all_anime_ids if anime_id not in rated_anime_ids]

    # Predict ratings for all unrated anime
    predictions = []
    for anime_id in unrated_anime_ids:
        prediction = model.predict(user_id, anime_id).est
        predictions.append((anime_id, prediction))

    # Sort predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top N recommendations
    top_recommendations = predictions[:n_recommendations]

    # Build a DataFrame for the recommended anime
    recommended_anime_ids = [anime_id for anime_id, _ in top_recommendations]
    recommended_anime = anime_df[anime_df["anime_id"].isin(recommended_anime_ids)].copy()

    # Add predicted scores to the DataFrame
    recommended_anime["predicted_score"] = recommended_anime["anime_id"].map(
        dict(top_recommendations)
    )

    # Sort by predicted score
    recommended_anime = recommended_anime.sort_values(by="predicted_score", ascending=False)

    return recommended_anime.reset_index(drop=True)
