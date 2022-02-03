from sklearn import preprocessing


# Normalization of supports by using standardization (mean=0, stddev=1)
class standardization:
    def __init__(self):
        self.transformer = preprocessing.StandardScaler(copy=False)
        self.mean = None
        self.scale = None

    def predict(self, data):
        return (data - self.mean) / self.scale #self.transformer.transform(data)

    def train(self, data):
        self.transformer.fit(data)
        self.mean = self.transformer.mean_.item()
        self.scale = self.transformer.scale_.item()