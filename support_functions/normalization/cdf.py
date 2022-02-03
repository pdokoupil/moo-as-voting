import numpy as np

from sklearn import preprocessing

# CDF normalization for the support functions
class cdf:
    def __init__(self):
        self.transformer = preprocessing.QuantileTransformer(copy=False)

    def predict(self, data):
        if data == self.lower_bound_x:
            return self.lower_bound_y
        elif data == self.upper_bound_x:
            return self.upper_bound_y
        return 0.5 * (np.interp(data, self.quantiles, self.references) - np.interp(-data, self.quantiles_neg_reversed, self.references_neg_reversed))

    def train(self, data):
        self.transformer.fit(data)
        
        self.quantiles = self.transformer.quantiles_[:, 0]
        self.quantiles_neg_reversed = -self.quantiles[::-1]
        
        self.references = self.transformer.references_
        self.references_neg_reversed = -self.references[::-1]

        self.lower_bound_x = self.quantiles[0]
        self.upper_bound_x = self.quantiles[-1]
        self.lower_bound_y = 0
        self.upper_bound_y = 1