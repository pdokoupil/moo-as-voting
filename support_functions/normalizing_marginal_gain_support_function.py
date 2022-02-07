import numpy as np

from recsys.recommendation_list import recommendation_list

class normalizing_marginal_gain_support_function:
    def __init__(self, objective, user):
        self.objective = objective
        self.user = user
        self.cache_key = None
        self.cache_value = None # Very simple cache with a size == 1
        self.cache_extra_value = None # Some temporary cache space available to the objectives
        self.normalization = None

    def set_normalization(self, normalization):
        self.normalization = normalization

    def __call__(self, item, context):
        recsys_list = context.get_current_recommendation_list_of_user(self.user)

        if recsys_list is None:
            recsys_list = recommendation_list(context.k, [])

        assert len(recsys_list) < recsys_list.k, \
            f"There still must be an empty space for the item {item} in the current recommendation list {recsys_list} of user {self.user}"
        
        # Caching optimization
        if self.cache_key != recsys_list.items:
            obj_value = self.objective(recsys_list, context) # TODO improve cache
            self.cache_value, self.cache_extra_value = self._get_obj_value(obj_value), self._get_cache_extra_value(obj_value)
            self.cache_key = recsys_list.items[:]
        
        # TODO mention optimization
        if hasattr(self.objective, "optimized_computation"):
            new_obj = self.objective.optimized_computation(recsys_list, item, context, self.cache_value, self.cache_extra_value)
            return self.normalization.predict(new_obj - self.cache_value)
        
        # Return obj(list + item) - obj(list)
        new_obj = self._get_obj_value(self.objective(recsys_list.with_extra_item(item), context))
        return self.normalization.predict(new_obj - self.cache_value)

    def _get_obj_value(self, obj_result):
        if isinstance(obj_result, tuple):
            return obj_result[0]
        return obj_result

    def _get_cache_extra_value(self, obj_result):
        if isinstance(obj_result, tuple):
            return obj_result[1]
        return None

    # TODO: handling uniqueness in the names of objectives/support functions
    def get_name(self):
        return self.objective.get_name()