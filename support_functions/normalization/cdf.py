import numpy as np
import os
from sklearn import preprocessing
from recsys.recommendation_list import recommendation_list

import pickle

# CDF normalization for the support functions
class cdf:
    def __init__(self, shift=0.0, cache_path=None): # Shift is usually 0.0 or -0.2
        print(f"Creating CDF with shift={shift}")
        self.shift = shift
        self.per_user = False

        self.quantiles = dict()
        self.quantiles_neg_reversed = dict()

        self.references = dict()
        self.references_neg_reversed = dict()

        self.lower_bound_x = dict()
        self.upper_bound_x = dict()

        self.cache_path = cache_path

        self.lower_bound_y = 0
        self.upper_bound_y = 1

    def predict(self, data, user):
        if self.per_user:
            if user in self.lower_bound_x:
                # Known user
                if data == self.lower_bound_x[user]:
                    return self.lower_bound_y + self.shift
                elif data == self.upper_bound_x[user]:
                    return self.upper_bound_y + self.shift
                return 0.5 * (np.interp(data, self.quantiles[user], self.references[user]) - np.interp(-data, self.quantiles_neg_reversed[user], self.references_neg_reversed[user])) + self.shift
            else:
                # Fallback for cases that we have per_user data, but the passed user is uknown        
                sampled_users = np.random.choice(list(self.lower_bound_x.keys()), 100)
                result = 0
                for sampled_user in sampled_users:
                    if data == self.lower_bound_x[sampled_user]:
                        result += self.lower_bound_y + self.shift
                    elif data == self.upper_bound_x[sampled_user]:
                        result += self.upper_bound_y + self.shift
                    else:
                        result += 0.5 * (np.interp(data, self.quantiles[sampled_user], self.references[sampled_user]) - np.interp(-data, self.quantiles_neg_reversed[sampled_user], self.references_neg_reversed[sampled_user])) + self.shift
                return result / len(sampled_users)
        else:
            if data == self.lower_bound_x:
                return self.lower_bound_y + self.shift
            elif data == self.upper_bound_x:
                return self.upper_bound_y + self.shift
            return 0.5 * (np.interp(data, self.quantiles, self.references) - np.interp(-data, self.quantiles_neg_reversed, self.references_neg_reversed)) + self.shift

    def train(self, users, lists, obj, ctx):
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading from cache: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                loaded_data = pickle.load(f)
                self.lower_bound_x = loaded_data.lower_bound_x
                self.upper_bound_x = loaded_data.upper_bound_x
                self.per_user = loaded_data.per_user
                self.quantiles = loaded_data.quantiles
                self.quantiles_neg_reversed = loaded_data.quantiles_neg_reversed
                self.references = loaded_data.references
                self.references_neg_reversed = loaded_data.references_neg_reversed
            return

        data = []
        transformers = dict()

        if not users:
            self.per_user = False
            for l in lists:
                data.append([obj(recommendation_list(ctx.k, list(l)), ctx)])
            transformers = preprocessing.QuantileTransformer(copy=False)
            transformers.fit(data)

            self.quantiles = transformers.quantiles_[:, 0]
            self.quantiles_neg_reversed = -self.quantiles[::-1]
            
            self.references = transformers.references_
            self.references_neg_reversed = -self.references[::-1]

            self.lower_bound_x = self.quantiles[0]
            self.upper_bound_x = self.quantiles[-1]
        else:
            self.per_user = True
            u_data = {}
            for u in users:
                u_data[u] = []
                for l in lists:
                    u_data[u].append([obj(u)(recommendation_list(ctx.k, list(l)), ctx)])
                
                transformers[u] = preprocessing.QuantileTransformer(copy=False)
                transformers[u].fit(u_data[u])

                self.quantiles[u] = transformers[u].quantiles_[:, 0]
                self.quantiles_neg_reversed[u] = -self.quantiles[u][::-1]
                
                self.references[u] = transformers[u].references_
                self.references_neg_reversed[u] = -self.references[u][::-1]

                self.lower_bound_x[u] = self.quantiles[u][0]
                self.upper_bound_x[u] = self.quantiles[u][-1]
        
        with open(self.cache_path, 'wb') as f:
            print(f"Saving cache to: {self.cache_path}")
            pickle.dump(self, f)