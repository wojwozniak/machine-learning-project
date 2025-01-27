import numpy as np


class KNNRecommender:

    def __init__(self):
        """
        Initialize the KNN Recommender model.

        Parameters:
        - k: Number of neighbors to consider (default is 5).
        - weight_factor: Adjusts the influence of weights in the weighted average.
                         A value of 1.0 uses the weights directly.
                         A value > 1.0 increases the impact of highly similar neighbors.
                         A value < 1.0 reduces the influence of weights.
        """

        self.users_base = None
        self.train_matrix = None

    def _cosine_similarity_filtered(self, mat, target_users, penalty_factor):
        """
        Custom cosine similarity calculation considering only non-zero values.
        Computes similarity between target_users and users_base.

        Parameters:
        - mat: Rating matrix.
        - target_users: Indices of users for whom similarity is computed.
        """
        similarity = np.zeros(
            (len(target_users), len(self.users_base)), dtype=np.float64
        )

        for idx, user in enumerate(target_users):
            for jdx, base_user in enumerate(self.users_base):
                valid_indices = (mat[user, :] > 0) & (mat[base_user, :] > 0)
                num_common = np.sum(valid_indices)
                if valid_indices.any():
                    user_ratings = mat[user, valid_indices]
                    base_user_ratings = mat[base_user, valid_indices]
                    numerator = np.dot(user_ratings, base_user_ratings)
                    denominator = np.sqrt(np.sum(user_ratings**2)) * np.sqrt(
                        np.sum(base_user_ratings**2)
                    )
                    if denominator > 0:
                        similarity[idx, jdx] = (numerator / denominator) * (
                            1 - penalty_factor / (num_common + 1e-9)
                        )
        return (similarity + 1 + penalty_factor) / (2 + penalty_factor)

    def fit(self, train_matrix, min_ratings_per_anime=20):
        """
        Fit the model using the training user-anime rating matrix and select users_base.

        Parameters:
        - train_matrix: A matrix where rows represent users, columns represent anime,
                        and values are ratings. Missing ratings should be filled with 0.
        - min_ratings_per_anime: Minimum number of ratings required for an anime
                                to include its users in users_base.
        """
        self.train_matrix = np.array(train_matrix)

        # Counter for each anime
        anime_ratings_count = np.zeros(len(train_matrix[0]))

        # Anime where counter is too low - propably every anime
        remaining_anime = np.where(anime_ratings_count < min_ratings_per_anime)[0]

        users_base_set = set()

        anime_counters = {
            anime: anime_ratings_count[anime]
            for anime in range(self.train_matrix.shape[1])
        }

        while len(remaining_anime) > 0:
            # Smallest number of raitings
            target_anime = remaining_anime[
                np.argmin([anime_counters[a] for a in remaining_anime])
            ]

            # Users who rated
            users_who_rated = np.where(self.train_matrix[:, target_anime] > 0)[0]

            if len(users_who_rated) > 0:
                # Most useful user (has many raitings)
                best_user = max(
                    users_who_rated, key=lambda u: np.sum(self.train_matrix[u] > 0)
                )

                users_base_set.add(best_user)
                user_anime_indices = np.where(self.train_matrix[best_user] > 0)[0]
                anime_counters[target_anime] -= 1
                for anime in user_anime_indices:
                    anime_counters[anime] += 1

            anime_counters[target_anime] += 1
            remaining_anime = np.array(
                [
                    anime
                    for anime in remaining_anime
                    if anime_counters[anime] < min_ratings_per_anime
                ]
            )

        self.users_base = list(users_base_set)
        return f"Selected {len(self.users_base)} users for users_base.\nSuccesful fit"

    def predict(self, user_indices, k=5, weight_factor=145, penalty_factor=0.8):
        """
        Predict ratings for specific users.

        Args:
            user_indices (list): List of user indices for whom to predict ratings.

        Returns:
            numpy.ndarray: A matrix with predicted ratings for the given users (dtype=float64).
        """
        if self.train_matrix is None:
            raise ValueError("The model must be fit before calling predict.")

        similarity = self._cosine_similarity_filtered(
            self.train_matrix, user_indices, penalty_factor
        )

        assert not np.any(np.isnan(similarity)), "Similarity matrix contains NaN!"
        assert not np.any(np.isinf(similarity)), "Similarity matrix contains Inf!"

        # Initialize an empty predictions matrix for the specified users
        predictions = np.zeros(
            (len(user_indices), self.train_matrix.shape[1]), dtype=np.float64
        )

        # Iterate over the specified user indices
        for idx, user in enumerate(user_indices):
            # Get the indices of the user's unrated anime (0 in the training matrix)
            unrated_anime_indices = np.where(self.train_matrix[user] == 0)[0]

            # We want to sort it once for each user
            similar_users = sorted(
                [
                    (self.users_base[jdx], similarity[idx, jdx])
                    for jdx in range(len(self.users_base))
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            for anime in range(len(self.train_matrix[user])):
                if anime in unrated_anime_indices:

                    # Filter users who rated this anime
                    rated_users = [
                        (user, sim)
                        for user, sim in similar_users
                        if self.train_matrix[user, anime] > 0
                    ]

                    # Take top K
                    top_k_users = rated_users[:k]

                    if not top_k_users:
                        print(
                            f"No top_k_users for user {user}, anime {anime}, rated_users {rated_users}"
                        )

                    # Compute weighted average of ratings for the anime
                    if top_k_users:
                        numer = sum(
                            (sim**weight_factor) * self.train_matrix[other_user, anime]
                            for other_user, sim in top_k_users
                        )
                        if np.isnan(numer):
                            print(
                                top_k_users,
                                weight_factor,
                                [
                                    (sim**weight_factor)
                                    * self.train_matrix[other_user, anime]
                                    for other_user, sim in top_k_users
                                ],
                            )
                        denom = sum(sim**weight_factor for _, sim in top_k_users)
                        predictions[idx, anime] = numer / (denom + 1e-9)
                else:
                    predictions[idx, anime] = self.train_matrix[user][anime]

        return predictions
