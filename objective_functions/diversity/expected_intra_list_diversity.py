import numpy as np

# Implementation of EILD metric/objective based on the following paper:
#
# S. Oliveira, V. Diniz, A. Lacerda and G. L. Pappa, "Multi-objective Evolutionary Rank Aggregation for Recommender Systems,"
# 2018 IEEE Congress on Evolutionary Computation (CEC), 2018, pp. 1-8, doi: 10.1109/CEC.2018.8477669.

class expected_intra_list_diversity:
    def _rd(self, k):
        return 0.85 ** (k - 1)

    def _similarity(self, item_i, item_j, context):
        u_i = context.dataset_statistics.users_viewed_item[item_i]
        u_j = context.dataset_statistics.users_viewed_item[item_j]
        return len(u_i.intersect(u_j)) / (np.sqrt(len(u_i)) * np.sqrt(len(u_j)))

    def _distance(self, item_i, item_j, context):
        return 1.0 - self._similarity(item_i, item_j, context)

    def __call__(self, recommendation_list, context, m=None):
        if m is None:
            m = len(recommendation_list)

        c = 0.0
        for i in range(m):
            c += self._rd(i + 1)

        eild_value = 0.0
        for i in range(m):
            c_k = 0.0
            for j in range(m):
                if i != j:
                    c_k += self._rd(max(1, j - i))
            c_k = c / c_k
            for j in range(m):
                if i != j:
                    eild_value += c_k * self._rd(i + 1) * self._rd(max(1, j - i)) * self._distance(recommendation_list[i], recommendation_list[j], context)

        assert eild_value <= 1.0, "EILD must be <= 1"
        return eild_value

    def get_name(self):
        return self.__class__.__name__