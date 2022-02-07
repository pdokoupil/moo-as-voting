import numpy as np

from sklearn import preprocessing
from recsys.recommendation_list import recommendation_list

SHIFT = 0.5
# CDF normalization for the support functions
class cdf:
    def __init__(self):
        self.transformer = preprocessing.QuantileTransformer(copy=False)

    def predict(self, data):
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
        else:
            for u in users:
                for l in lists:
                    data.append([obj(u)(recommendation_list(ctx.k, list(l)), ctx)])

        self.transformer.fit(data)

        self.quantiles = self.transformer.quantiles_[:, 0]
        self.quantiles_neg_reversed = -self.quantiles[::-1]
        
        self.references = self.transformer.references_
        self.references_neg_reversed = -self.references[::-1]

        self.lower_bound_x = self.quantiles[0]
        self.upper_bound_x = self.quantiles[-1]
        self.lower_bound_y = 0
        self.upper_bound_y = 1