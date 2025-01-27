import pickle
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from threadpoolctl import threadpool_limits
import os
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained model and mappings
try:
    with open("./backend/als_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("./backend/anime_id_to_index.pkl", "rb") as f:
        anime_id_to_index = pickle.load(f)
    with open("./backend/user_id_to_index.pkl", "rb") as f:
        user_id_to_index = pickle.load(f)
except Exception as e:
    logger.error(f"Error loading model or mappings: {e}")
    raise

# Load anime data
anime = pd.read_csv("./data/anime.csv")

# Create reverse mapping for anime IDs
index_to_anime_id = {idx: anime_id for anime_id,
                     idx in anime_id_to_index.items()}


def get_recommendations_for_new_user(rating_list, model, anime_df, anime_id_to_index, n_recommendations=10):
    """
    Generate recommendations for a new user based on their ratings.
    """
    n_items = len(anime_id_to_index)
    data = []
    cols = []

    for anime_id, rating in rating_list:
        if anime_id in anime_id_to_index:
            idx = anime_id_to_index[anime_id]
            data.append(1 + np.log1p(rating))
            cols.append(idx)
        else:
            logger.warning(f"Anime ID {anime_id} not found in mappings.")

    new_user_ratings = csr_matrix(
        (data, (np.zeros_like(cols), cols)),
        shape=(1, n_items)
    )

    ids, scores = model.recommend(
        userid=0,
        user_items=new_user_ratings,
        N=n_recommendations,
        filter_already_liked_items=True,
        recalculate_user=True
    )

    recommended_anime_ids = [index_to_anime_id[idx] for idx in ids]
    recommended_anime = anime_df[anime_df['anime_id'].isin(
        recommended_anime_ids)].copy()
    score_dict = dict(zip(recommended_anime_ids, scores))
    recommended_anime['score'] = recommended_anime['anime_id'].map(score_dict)

    result = recommended_anime[['name', 'genre', 'type', 'rating', 'score']].sort_values(
        by='score', ascending=False
    )

    return result


@app.route('/recommendations', methods=['POST'])
def recommend():
    """
    Endpoint to get anime recommendations based on user ratings.
    Expects a JSON payload with a 'ratings' key containing a list of (anime_id, rating) tuples.
    """
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")

        if not data or 'ratings' not in data:
            return jsonify({"error": "No ratings provided"}), 400

        rating_list = data['ratings']

        # Validate ratings
        if not isinstance(rating_list, list) or not all(isinstance(r, (list, tuple)) and len(r) == 2 for r in rating_list):
            return jsonify({"error": "Invalid ratings format. Expected a list of (anime_id, rating) tuples."}), 400

        recommendations = get_recommendations_for_new_user(
            rating_list, model, anime, anime_id_to_index
        )

        return jsonify(recommendations.to_dict(orient="records"))

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
