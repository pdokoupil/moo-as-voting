class precision:
    def __init__(self, user):
        self.user = user

    # Calculates Precision@m
    def __call__(self, recommendation_list, context, m=None):
        if m is None:
            m = len(recommendation_list)
        
        return len(context.relevant_items[self.user].intersection(recommendation_list[:m])) / len(recommendation_list[:m])

    def get_name(self):
        return self.__class__.__name__