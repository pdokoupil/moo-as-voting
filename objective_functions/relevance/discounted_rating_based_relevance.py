import numpy as np

class discounted_rating_based_relevance:
    def __init__(self, user, rating_matrix, discount):
        self.user = user
        self.rating_matrix = rating_matrix
        self.discount = discount

    def is_discounted(self):
        return True

    def get_discount(self, k):
        return self.discount ** k

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
        for k, item in enumerate(recommendation_list.items[:m]):
            item_id = context.recsys_statistics.item_to_item_id[item]
            ratings += self.get_discount(k) *rating_row[item_id]

        return ratings #/ n

    # Special case when the size of recommendation list is 0
    def _special_case_size_0(self, recommendation_list, context):
        # Take average rating relevance over all items with NON-ZERO relevance
        return 0 #self.avg_rating

    def optimized_computation(self, old_recommendation_list, new_item, context, old_recommendation_list_value, cache_extra_value, m=None):
        # Assume that new_recommendation_list = old_recommendation_list + [new_item] and that we have
        # objective value for the old_recommendation_list
        # We want to reuse as much information from the old_recommendation_list to compute objective for the new list
        # more efficiently. This optimization is objective specific (i.e. some objecrives can be optimized easily while others cannot be optimized at all)
        # n = len(old_recommendation_list)
        user_id = context.recsys_statistics.user_to_user_id[self.user]
        item_id = context.recsys_statistics.item_to_item_id[new_item]
        if user_id >= self.rating_matrix.shape[0]:
            # Approximate the rating with average
            rating = self.rating_matrix[:, item_id].mean()
        else:
            rating = self.rating_matrix[user_id, item_id]
        #return ((old_recommendation_list_value * n) + rating) / (n + 1)
        return old_recommendation_list_value + self.get_discount(len(old_recommendation_list)) * rating

    def get_name(self):
        return self.__class__.__name__