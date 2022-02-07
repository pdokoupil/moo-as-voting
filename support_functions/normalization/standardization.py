from sklearn import preprocessing
from recsys.recommendation_list import recommendation_list

# Normalization of supports by using standardization (mean=0, stddev=1)
class standardization:
    def __init__(self):
        self.transformer = preprocessing.StandardScaler(copy=False)
        self.mean = None
        self.scale = None

    def predict(self, data):
        return (data - self.mean) / self.scale #self.transformer.transform(data)

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
        self.mean = self.transformer.mean_.item()
        self.scale = self.transformer.scale_.item()