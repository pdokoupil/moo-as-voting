from sklearn import preprocessing
from recsys.recommendation_list import recommendation_list
import pickle
import os
import numpy as np

# Normalization of supports by using standardization (mean=0, stddev=1)
class standardization:
    def __init__(self, shift=0.0, cache_path=None): # shift is usually 0.0 or 2.0
        print(f"Creating standardization with shift={shift}")
        self.shift = shift
        self.per_user = False

        self.means = dict()
        self.scales = dict()

        self.cache_path = cache_path

    def predict(self, data, user):
        if self.per_user:
            if user in self.means:
                # Known user
                return (data - self.means[user]) / self.scales[user] + self.shift
            else:
                # Fallback for cases that we have per_user data, but the passed user is uknown        
                sampled_users = np.random.choice(list(self.means.keys()), 100)
                result = 0
                for sampled_user in sampled_users:
                    result += (data - self.means[sampled_user]) / self.scales[sampled_user] + self.shift
                return result / len(sampled_users)
        return (data - self.means) / self.scales + self.shift

    def train(self, users, lists, obj, ctx):
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading from cache: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                loaded_data = pickle.load(f)
                self.means = loaded_data.means
                self.scales = loaded_data.scales
                self.per_user = loaded_data.per_user
            return

        if not users:
            self.per_user = False
            data = []
            for l in lists:
                data.append([obj(recommendation_list(ctx.k, list(l)), ctx)])
            
            transformer = preprocessing.StandardScaler(copy=False)
            transformer.fit(data)
            
            self.means = transformer.mean_.item()
            self.scales = transformer.scale_.item()
        else:
            self.per_user = True
            u_data = dict()
            for u in users:
                u_data[u] = []
                for l in lists:
                    u_data[u].append([obj(u)(recommendation_list(ctx.k, list(l)), ctx)])
            
                transformer = preprocessing.StandardScaler(copy=False)
                transformer.fit(u_data[u])

                self.means[u] = transformer.mean_.item()
                self.scales[u] = transformer.scale_.item()
                
        with open(self.cache_path, 'wb') as f:
            print(f"Saving cache to: {self.cache_path}")
            pickle.dump(self, f)