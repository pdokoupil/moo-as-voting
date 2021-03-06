import numpy as np
from itertools import combinations

from recsys.recommendation_list import recommendation_list

class intra_list_diversity:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        
    def is_discounted(self):
        return False

    def __call__(self, recommendation_list, context, m=None):
        n = len(recommendation_list.items)

        # # Handle corner/special cases
        if n == 0:
            return self._special_case_size_0(recommendation_list, context)
        elif n == 1:
            return self._special_case_size_1(recommendation_list, context)
        
        if m is None:
            m = n
        
        # div = sum(
        #     map( # We are exploiting symmetry here
        #         lambda x: 2.0 * (1.0 - self.similarity_matrix[self._get_id(x[0], context), self._get_id(x[1], context)]), # TODO exploit symmetry
        #         combinations(recommendation_list.items[:m], 2)
        #     )
        # ) / n #(n * (n - 1))

        #np.apply_along_axis(lambda x: context.recsys_statistics.item_to_item_id[x[0]], 1, np.expand_dims(recommendation_list.items, 1))
        top_k_list = [context.recsys_statistics.item_to_item_id[i] for i in recommendation_list.items]
        div = self.distance_matrix[np.ix_(top_k_list, top_k_list)].sum() / n

        return div

    def get_name(self):
        return self.__class__.__name__

    # Special case when the size of recommendation list is 0
    def _special_case_size_0(self, recommendation_list, context):
        # Take average diversity between pair of items (averaged over all pairs)
        # assert self.similarity_matrix.shape[0] == self.similarity_matrix.shape[1]
        # n = self.similarity_matrix.shape[0] * self.similarity_matrix.shape[0] - self.similarity_matrix.shape[0] # Skip the diagonal elements
        # return (1.0 - self.similarity_matrix).sum() / n
        return 0

    def _special_case_size_1(self, recommendation_list, context):
        # Take average diversity of (recommendation_list[0], item) where item is arbitrary item from the dataset
        item_id = self._get_id(recommendation_list[0], context)
        # n = 0
        # div = 0
        # for i in range(self.similarity_matrix.shape[0]):
        #     if i != item_id:
        #         div += 1.0 - self.similarity_matrix[item_id, i]
        #         n += 1
        # return div / n

        # TODO: multiply the sum by 2?
        return self.distance_matrix[item_id].sum() / (self.distance_matrix.shape[0] - 1) # Subtract 1 for sim(item_id, item_id)
        
    def optimized_computation(self, old_recommendation_list, new_item, context, old_recommendation_list_value, cache_extra_value, m=None):
        # Assume that new_recommendation_list = old_recommendation_list + [new_item] and that we have
        # objective value for the old_recommendation_list
        # We want to reuse as much information from the old_recommendation_list to compute objective for the new list
        # more efficiently. This optimization is objective specific (i.e. some objecrives can be optimized easily while others cannot be optimized at all)
        n = len(old_recommendation_list.items)
        
        # So now we are computing for recommendation list with size 1 which is handled as a special case
        if n == 0:
            #return self._special_case_size_1(old_recommendation_list.with_extra_item(new_item), context)
            return self(old_recommendation_list.with_extra_item(new_item), context)
        elif n == 1:
            # We have list of length 1 and we add 1 more item .. there is nothing we can optimize
            return self(old_recommendation_list.with_extra_item(new_item), context)
        
        # s = 0.0
        j_id = self._get_id(new_item, context)
        # for item in old_recommendation_list:
        #     i_id = self._get_id(item, context)
        #     s += 2.0 * self.distance_matrix[i_id, j_id] #(1.0 - self.similarity_matrix[i_id, j_id]) # We are exploiting symmetry here
        
        top_k_list = np.array([*map(context.recsys_statistics.item_to_item_id.get, old_recommendation_list.items)]) #[context.recsys_statistics.item_to_item_id[i] for i in old_recommendation_list]
        s = self.distance_matrix[j_id, top_k_list].sum() * 2
        return ((old_recommendation_list_value * n) + s) / (n + 1) #((old_recommendation_list_value * n * (n - 1)) + s) / ((n + 1) * n)

    def _get_id(self, x, ctx):
        return ctx.recsys_statistics.item_to_item_id[x]