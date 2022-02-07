from sklearn import preprocessing
from recsys.recommendation_list import recommendation_list

# Normalization of supports by using robut standardization
class robust_scaling:
    def __init__(self):
        self.transformer = preprocessing.RobustScaler(copy=False)
        self.center = None
        self.scale = None

    def predict(self, data):
        return (data - self.center) / self.scale #self.transformer.transform(data)

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
        self.center = self.transformer.center_.item()
        self.scale = self.transformer.scale_.item()