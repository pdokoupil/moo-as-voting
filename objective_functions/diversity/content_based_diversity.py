class content_based_diversity:
    def __init__(self, metadata):
        self.metadata = metadata
        genres = set()
        for m in self.metadata.values():
            genres.update(m["genres"])
        self.num_genres = len(genres)

    def is_discounted(self):
        return False

    def __call__(self, recommendation_list, context, m=None):
        n = len(recommendation_list.items)

        if m is None:
            m = n
        
        genres_in_list = self._genres_in_list(recommendation_list.items[:m])

        return len(genres_in_list) / self.num_genres#, genres_in_list

    def _genres_in_list(self, recommendation_list_items):
        genres_present = set()
        for i in recommendation_list_items:
            if i in self.metadata:
                genres_present.update(self.metadata[i]["genres"])
        return genres_present

    def get_name(self):
        return self.__class__.__name__

    def optimized_computation(self, old_recommendation_list, new_item, context, old_recommendation_list_value, cache_extra_value, m=None):
        
        if new_item in self.metadata:
            new_item_metadata = self.metadata[new_item]["genres"]
        else:
            new_item_metadata = set()

        old_recommendation_list_metadata = self._genres_in_list(old_recommendation_list)

        #return old_recommendation_list_value + len(new_item_metadata) - cache_extra_value.intersection(new_item_metadata)
        return old_recommendation_list_value + len(new_item_metadata) - len(old_recommendation_list_metadata.intersection(new_item_metadata))