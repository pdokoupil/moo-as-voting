# Inverse popularity / popularity complement novelty metric
class popularity_complement:
    def __call__(self, recommendation_list, context, m=None):
        if m is None:
            m = len(recommendation_list)

        if m == 0:
            return self._special_case_size_0(recommendation_list, context)

        nov = 0.0
        n = 0
        for i, item in enumerate(recommendation_list[:m]):
            nov += (1.0 - self._popularity(item, context))
            n += 1

        return nov / n

    def _popularity(self, item, context):
            return len(context.dataset_statistics.users_viewed_item[item]) / context.num_users

    # Special case when the size of recommendation list is 0
    def _special_case_size_0(self, recommendation_list, context):
        nov = 0
        n = 0
        for item, _ in context.recsys_statistics.item_to_item_id.items():
            nov += (1.0 - self._popularity(item, context))
            n += 1

        return  nov / n

    def optimized_computation(self, old_recommendation_list, new_item, context, old_recommendation_list_value, cache_extra_value, m=None):
        # Assume that new_recommendation_list = old_recommendation_list + [new_item] and that we have
        # objective value for the old_recommendation_list
        # We want to reuse as much information from the old_recommendation_list to compute objective for the new list
        # more efficiently. This optimization is objective specific (i.e. some objecrives can be optimized easily while others cannot be optimized at all)
        n = len(old_recommendation_list)
        return ((old_recommendation_list_value * n) + (1.0 - self._popularity(new_item, context))) / (n + 1)

    def get_name(self):
        return self.__class__.__name__