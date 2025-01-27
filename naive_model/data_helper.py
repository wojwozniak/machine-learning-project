import pandas as pd
import numpy as np


def get_data():

    # get data
    file_path = "../data"
    anime = pd.read_csv(file_path + "/anime.csv")
    ratings = pd.read_csv(file_path + "/rating.csv")

    # drop rows with missing values in 'genre' or 'rating' and missing IDs
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

    # remove IDs
    ratings.drop(["anime_id", "user_id"], axis=1, inplace=True)

    return ratings


def data_to_matrix(data):
    # remove duplicates
    ratings = data.groupby(["user_idx", "anime_idx"], as_index=False).mean()

    # create user-anime matrix
    user_anime_matrix = ratings.pivot(
        index="user_idx", columns="anime_idx", values="rating"
    ).fillna(0)

    return user_anime_matrix.values
