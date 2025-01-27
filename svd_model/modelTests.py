import pandas as pd
import numpy as np
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import  GridSearchCV, cross_validate
from surprise import Reader, Dataset
from collections import defaultdict
import random
from surprise import accuracy

ratings = "../data/rating.csv"
anime = "../data/anime.csv"

# Remove unrated anime
df_ratings = pd.read_csv(ratings, sep=",", names=["user_id" ,"anime_id" ,"rating"], skiprows=1)
df_anime = pd.read_csv(anime, sep=",", names=["anime_id","name", "genre","type","episodes","rating","members"], skiprows=1)

# Clean data
df_anime = df_anime.dropna(subset=["genre", "rating"])
df_ratings = df_ratings.dropna(subset=["anime_id", "user_id"])
df_anime = df_anime.dropna(subset=["anime_id"])
df_ratings = df_ratings[df_ratings["rating"] != -1]

# Ensure we only include anime_ids that exist in the anime dataset
valid_anime_ids = set(df_anime["anime_id"].unique())
df_ratings = df_ratings[df_ratings["anime_id"].isin(valid_anime_ids)]

reader = Reader(rating_scale=(1,10))

def custom_train_test_split(df, min_ratings=4, test_ratio=0.2):
    """
    Splits data into a training and testing set, ensuring each user in the test set
    has at least `min_ratings` - 1 ratings in the training set.

    Args:
        df: Pandas DataFrame containing 'user', 'item', and 'rating' columns.
        min_ratings: Minimum number of ratings a user must have to be considered.
        test_ratio: Proportion of each user's ratings to include in the test set.

    Returns:
        trainset: Surprise trainset object.
        testset: List of (user, item, true_rating) tuples for testing.
    """

    # Group ratings by user
    user_ratings = defaultdict(list)
    for _, row in df.iterrows():
        user_ratings[row["user_id"]].append((row["anime_id"], row["rating"]))
    
    train_ratings = []
    test_ratings = []

    for uid, ratings in user_ratings.items():
        # Skip users with fewer than `min_ratings` ratings and just add them to training set
        if len(ratings) < min_ratings:
            train_ratings.extend([(uid, iid, rating) for iid, rating in ratings])
            continue
        
        # Split into train and test
        num_test = max(1, int(len(ratings) * test_ratio))
        test_indices = random.sample(range(len(ratings)), num_test)

        train_count = len(ratings) - num_test

        for i, (iid, rating) in enumerate(ratings):
            if i in test_indices:
                test_ratings.append((uid, iid, rating, train_count))
            else:
                train_ratings.append((uid, iid, rating))
    
    return train_ratings, test_ratings

# Perform the custom split
train_raitings, testset = custom_train_test_split(df_ratings, min_ratings=4, test_ratio=0.2)

# Build trainset
trainset = Dataset.load_from_df(
    pd.DataFrame(train_raitings, columns=["user_id", "anime_id", "rating"]),
    reader,
).build_full_trainset()

# Example: Train your model on the trainset
svd = SVD(n_factors=100, reg_all=0.05)
svd.fit(trainset)

# Evaluate on the testset
predictions = [svd.predict(uid, iid, r_ui) for (uid, iid, r_ui, train_count) in testset]
print("RMSE:", accuracy.rmse(predictions))

predictions = [(svd.predict(uid, iid, r_ui),(train_count)) for (uid, iid, r_ui, train_count) in testset]

# Write predictions to a CSV file
## add ilosc treningowych ratingow vs testing
predictions_df = pd.DataFrame(
    [
        {
            "user": pred.uid,
            "item": pred.iid,
            "true_rating": pred.r_ui,
            "predicted_rating": pred.est,
            "details": pred.details,
            "train_count": train_count,
        }
        for (pred,train_count) in predictions
    ]
)

# Save to file
output_file = "./predictions.csv"
predictions_df.to_csv(output_file, index=False)

print("Predictions succesfully written to predictions")


