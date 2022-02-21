from recsys.recommendation_list import recommendation_list

class normalizing_marginal_gain_support_function:
    def __init__(self, objective, user, normalized_support_cache):
        self.objective = objective
        self.user = user
        self.cache_key = None
        self.cache_value = None # Very simple cache with a size == 1
        self.cache_extra_value = None # Some temporary cache space available to the objectives
        self.normalization = None
        
        # This cache is already for the given objective and normalization method
        self.obj_name = self.objective.get_name()
        self.normalized_support_cache = normalized_support_cache[self.obj_name]

        self.user_bound = hasattr(self.objective, "user")

    def set_normalization(self, normalization):
        self.normalization = normalization

    def __call__(self, item, context):
        recsys_list = context.get_current_recommendation_list_of_user(self.user)

        if recsys_list is None:
            recsys_list = recommendation_list(context.k, [])

        key = (tuple(recsys_list.items), item)

        if not self.user_bound:
            # TODO: DO NOT PASS User if not hasatttr(obj, user). Also add prediction cache to normalization
            if key in self.normalized_support_cache:
                return self.normalized_support_cache[key]

        if self.cache_key != recsys_list.items:
            obj_value = self.objective(recsys_list, context) # TODO improve cache
            self.cache_value, self.cache_extra_value = self._get_obj_value(obj_value), self._get_cache_extra_value(obj_value)
            self.cache_key = recsys_list.items[:]
        
        # TODO mention optimization
        if hasattr(self.objective, "optimized_computation"):
            new_obj = self.objective.optimized_computation(recsys_list, item, context, self.cache_value, self.cache_extra_value)
            
            if self.objective.is_discounted():
                discount = self.objective.get_discount(len(recsys_list.items))
                inverse_discount = 1.0 / discount
                sup = discount * self.normalization.predict((new_obj - self.cache_value) * inverse_discount, self.user) #discount(normalization(inverse_discount(O(L+)) - inverse_discount(O(L))))
            else:
                sup = self.normalization.predict(new_obj - self.cache_value, self.user)
            
            if not self.user_bound and len(recsys_list.items) < 2:
                self.normalized_support_cache[key] = sup
            
            return sup
        
        # Return obj(list + item) - obj(list)
        new_obj = self._get_obj_value(self.objective(recsys_list.with_extra_item(item), context))

        if self.objective.is_discounted():
            discount = self.objective.get_discount(len(recsys_list.items))
            inverse_discount = 1.0 / discount
            sup = discount * self.normalization.predict((new_obj - self.cache_value) * inverse_discount, self.user) #discount(normalization(inverse_discount(O(L+)) - inverse_discount(O(L))))
        else:
            sup = self.normalization.predict(new_obj - self.cache_value, self.user)
        
        if not self.user_bound and len(recsys_list.items) < 2:
            self.normalized_support_cache[key] = sup
        
        return sup

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