from model import user_id_to_index, anime_id_to_index, als_model, anime, interaction_matrix

# function to recommend anime for a given user


def recommend_anime(user_id, model, interaction_matrix, anime_df, n_recommendations=10):
    # convert user_id to internal user index
    user_idx = user_id_to_index[user_id]

    # get user recommendations (anime indices and scores)
    recommendations = model.recommend(
        user_idx,
        interaction_matrix[user_idx],
        N=n_recommendations,
        filter_already_liked_items=True,
    )

    # map anime indices back to anime_id
    anime_indices, scores = zip(*recommendations)
    recommended_anime_ids = [list(anime_id_to_index.keys())[
        i] for i in anime_indices]

    # retrieve anime metadata
    recommended_anime = anime[anime["anime_id"].isin(recommended_anime_ids)]
    recommended_anime["score"] = scores

    return recommended_anime[["name", "genre", "type", "rating", "score"]].sort_values(
        by="score", ascending=False
    )


# example: Get recommendations for user_id 1
user_id = 1
recommendations = recommend_anime(
    user_id, als_model, interaction_matrix, anime)
print("Recommendations for User {}:".format(user_id))
print(recommendations)
