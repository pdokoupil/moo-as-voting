import numpy as np

from objective_functions.relevance.precision import precision

# Note: https://stackoverflow.com/questions/55748792/understanding-precisionk-apk-mapk (TODO)

class average_precision:
    def __init__(self, user):
        self.user = user
        self.prec = precision(self.user)

    def __call__(self, recommendation_list, context, m=None):
        n = len(recommendation_list)
        if m is None:
            m = n
        
        # p = 0.0
        # for i in range(1, m + 1):
        #     p += (self.prec(recommendation_list, context, i)) / min(len(context.relevant_items[self.user]), n)

        # return p

        return np.mean([self.prec(recommendation_list, context, i) for i in range(1, m + 1)])

    def get_name(self):
        return self.__class__.__name__