from sklearn import preprocessing
import numpy as np

# MinMaxScaling normalization for the support values
class min_max_scaling:
    def __init__(self):
        self.transformer = preprocessing.MinMaxScaler(copy=False)
        self.min = None
        self.scale = None

    def predict(self, data):
        return data * self.scale + self.min #self.transformer.transform([[data]])

    def train(self, data):
        self.transformer.fit(data)
        self.scale = self.transformer.scale_.item()
        self.min = self.transformer.min_.item()