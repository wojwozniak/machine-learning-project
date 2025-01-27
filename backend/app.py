import pickle
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sparse
from threadpoolctl import threadpool_limits
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open("./backend/als_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("./backend/anime_id_to_index.pkl", "rb") as f:
    anime_id_to_index = pickle.load(f)

with open("./backend/user_id_to_index.pkl", "rb") as f:
    user_id_to_index = pickle.load(f)

anime = pd.read_csv("./data/anime.csv")

index_to_anime_id = {idx: anime_id for anime_id,
                     idx in anime_id_to_index.items()}


def get_recommendations_for_new_user(rating_list, model, anime_df, anime_id_to_index, n_recommendations=10):
    n_items = len(anime_id_to_index)
    data = []
    cols = []

    for anime_id, rating in rating_list:
        if anime_id in anime_id_to_index:
            idx = anime_id_to_index[anime_id]
            data.append(1 + np.log1p(rating))
            cols.append(idx)

    new_user_ratings = sparse.csr_matrix(
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
    try:
        data = request.get_json()
        print(data)
        rating_list = data['ratings']

        if not rating_list:
            return jsonify({"error": "No ratings provided"}), 400

        anime_tuples = data.get('animeTuples', [])
        print('Received anime tuples:', anime_tuples)

        recommendations = get_recommendations_for_new_user(
            rating_list, model, anime, anime_id_to_index
        )

        return jsonify(recommendations.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
