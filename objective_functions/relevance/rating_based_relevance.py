import numpy as np

class rating_based_relevance:
    def __init__(self, user, rating_matrix):
        self.user = user
        self.rating_matrix = rating_matrix

        # Skip zero (uknown) ratings
        nonzero = self.rating_matrix[np.nonzero(self.rating_matrix)]
        self.avg_rating = nonzero.sum() / nonzero.size

        self.min_rating = self.rating_matrix[self.rating_matrix > 0].min() * 0.1

    def __call__(self, recommendation_list, context, m=None):
        n = len(recommendation_list.items)
        if m is None:
            m = n

        if n == 0:
            return self._special_case_size_0(recommendation_list, context)

        user_id = context.recsys_statistics.user_to_user_id[self.user]
        
        if user_id >= self.rating_matrix.shape[0]:
            # Approximate the ratings for the user with movie rating average
            rating_row = self.rating_matrix.mean(axis=0)
        else:
            rating_row = self.rating_matrix[user_id]
        
        ratings = 0.0
        for item in recommendation_list.items[:m]:
            item_id = context.recsys_statistics.item_to_item_id[item]
            if rating_row[item_id] > 0:
                ratings += rating_row[item_id]
            else:
                ratings += self.min_rating

        return ratings / n

    # Special case when the size of recommendation list is 0
    def _special_case_size_0(self, recommendation_list, context):
        # Take average rating relevance over all items with NON-ZERO relevance
        return self.avg_rating

    def optimized_computation(self, old_recommendation_list, new_item, context, old_recommendation_list_value, cache_extra_value, m=None):
        # Assume that new_recommendation_list = old_recommendation_list + [new_item] and that we have
        # objective value for the old_recommendation_list
        # We want to reuse as much information from the old_recommendation_list to compute objective for the new list
        # more efficiently. This optimization is objective specific (i.e. some objecrives can be optimized easily while others cannot be optimized at all)
        n = len(old_recommendation_list)
        user_id = context.recsys_statistics.user_to_user_id[self.user]
        item_id = context.recsys_statistics.item_to_item_id[new_item]
        if user_id >= self.rating_matrix.shape[0]:
            # Approximate the rating with average
            rating = self.rating_matrix[:, item_id].mean()
        else:
            rating = self.rating_matrix[user_id, item_id]
        return ((old_recommendation_list_value * n) + rating) / (n + 1)

    def get_name(self):
        return self.__class__.__name__