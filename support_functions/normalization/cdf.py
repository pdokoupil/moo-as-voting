import numpy as np

from sklearn import preprocessing
from recsys.recommendation_list import recommendation_list

SHIFT = 0.2
# CDF normalization for the support functions
class cdf:
    def __init__(self):
        self.transformers = dict()

        self.quantiles = dict()
        self.quantiles_neg_reversed = dict()

        self.references = dict()
        self.references_neg_reversed = dict()

        self.lower_bound_x = dict()
        self.upper_bound_x = dict()

    def predict(self, data, user):
        if type(self.transformers) is dict:
            if data == self.lower_bound_x[user]:
                return self.lower_bound_y - SHIFT
            elif data == self.upper_bound_x[user]:
                return self.upper_bound_y - SHIFT
            return 0.5 * (np.interp(data, self.quantiles[user], self.references[user]) - np.interp(-data, self.quantiles_neg_reversed[user], self.references_neg_reversed[user])) - SHIFT
        else:
            if data == self.lower_bound_x:
                return self.lower_bound_y - SHIFT
            elif data == self.upper_bound_x:
                return self.upper_bound_y - SHIFT
            return 0.5 * (np.interp(data, self.quantiles, self.references) - np.interp(-data, self.quantiles_neg_reversed, self.references_neg_reversed)) - SHIFT

    def train(self, users, lists, obj, ctx):
        data = []

        if not users:
            for l in lists:
                data.append([obj(recommendation_list(ctx.k, list(l)), ctx)])
            self.transformers = preprocessing.QuantileTransformer(copy=False)
            self.transformers.fit(data)

            self.quantiles = self.transformers.quantiles_[:, 0]
            self.quantiles_neg_reversed = -self.quantiles[::-1]
            
            self.references = self.transformers.references_
            self.references_neg_reversed = -self.references[::-1]

            self.lower_bound_x = self.quantiles[0]
            self.upper_bound_x = self.quantiles[-1]
        else:
            u_data = {}
            for u in users:
                u_data[u] = []
                for l in lists:
                    u_data[u].append([obj(u)(recommendation_list(ctx.k, list(l)), ctx)])
                
                self.transformers[u] = preprocessing.QuantileTransformer(copy=False)
                self.transformers[u].fit(u_data[u])

                self.quantiles[u] = self.transformers[u].quantiles_[:, 0]
                self.quantiles_neg_reversed[u] = -self.quantiles[u][::-1]
                
                self.references[u] = self.transformers[u].references_
                self.references_neg_reversed[u] = -self.references[u][::-1]

                self.lower_bound_x[u] = self.quantiles[u][0]
                self.upper_bound_x[u] = self.quantiles[u][-1]


        self.lower_bound_y = 0
        self.upper_bound_y = 1