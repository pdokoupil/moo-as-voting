from sklearn import preprocessing
from recsys.recommendation_list import recommendation_list
import numpy as np

# MinMaxScaling normalization for the support values
class min_max_scaling:
    def __init__(self):
        self.transformer = preprocessing.MinMaxScaler(copy=False)
        self.min = None
        self.scale = None

    def predict(self, data):
        return data * self.scale + self.min #self.transformer.transform([[data]])

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
        self.scale = self.transformer.scale_.item()
        self.min = self.transformer.min_.item()