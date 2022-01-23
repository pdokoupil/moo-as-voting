
from objective_functions.relevance.average_precision import average_precision


class mean_average_precision:
    def __init__(self, user):
        self.user = user

    def __call__(self, recommendation_list, context, m=None):

        p = 0.0
        num_users =  0
        for user, recommendation_list in context.get_current_recommendation_lists().items():
            if m is None:
                user_m = len(recommendation_list)
            else:
                user_m = m

            avg_prec = average_precision(user)
            p += avg_prec(context, user_m)
            num_users += 1

        p /= num_users
        assert p <= 1.0, "MAP cannot exceed 1.0"
        return p

    def get_name(self):
        return self.__class__.__name__