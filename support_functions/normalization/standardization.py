from sklearn import preprocessing
from recsys.recommendation_list import recommendation_list

STANDARDIZATION_SHIFT = 2.0

# Normalization of supports by using standardization (mean=0, stddev=1)
class standardization:
    def __init__(self):
        self.transformers = dict() #preprocessing.StandardScaler(copy=False)
        self.means = dict()
        self.scales = dict()

    def predict(self, data, user):
        if type(self.transformers) is dict:
            return (data - self.means[user]) / self.scales[user] + STANDARDIZATION_SHIFT
        return (data - self.means) / self.scales + STANDARDIZATION_SHIFT

    def train(self, users, lists, obj, ctx):
        if not users:
            data = []
            for l in lists:
                data.append([obj(recommendation_list(ctx.k, list(l)), ctx)])
            
            self.transformers = preprocessing.StandardScaler(copy=False)
            self.transformers.fit(data)
            
            self.means = self.transformers.mean_.item()
            self.scales = self.transformers.scale_.item()
        else:
            u_data = dict()
            for u in users:
                u_data[u] = []
                for l in lists:
                    u_data[u].append([obj(u)(recommendation_list(ctx.k, list(l)), ctx)])
            
                self.transformers[u] = preprocessing.StandardScaler(copy=False)
                self.transformers[u].fit(u_data[u])

                self.means[u] = self.transformers[u].mean_.item()
                self.scales[u] = self.transformers[u].scale_.item()