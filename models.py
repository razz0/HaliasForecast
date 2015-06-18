"""Prediction models for Halias migration prediction based on weather forecast"""

class PredictionModel(object):
    """
    Abstract model class
    """

    def __init__(self, name, json_file, **model_kwargs):
        self.name = name
        self.JSON_FILE = json_file
        self.model_kwargs = model_kwargs
        self.disruptions = {}
        self.stored_disruptions = {}
        self.model = None  # Prediction model

    def predict(self, *args):
        return 0

    def generate_model(self, x, y):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass


class Model0(PredictionModel):
    '''Baseline model (always 0)'''

    def predict(self, *args):
        return 0

    def generate_model(self, x, y):
        from sklearn.neighbors import KNeighborsRegressor

        self.model = KNeighborsRegressor(**self.model_kwargs)
        self.model.fit(x, y)


class ScikitPredictor(PredictionModel):
    """Pre-calculated Scikit-learn prediction model"""

    def __init__(self, name, json_file, model_file, **model_kwargs):
        super(ScikitPredictor, self).__init__(name, json_file, **model_kwargs)
        self.model = None
        self.filename = model_file

    def predict(self, *args):
        val = self.model.predict(*args)
        if type(val) == list:
            if len(val) > 1:
                raise Exception('Predicted multiple values')
            return val[0]
        else:
            return int(val)

    def save_model(self):
        from sklearn.externals import joblib
        joblib.dump(self.model, self.filename)

    def load_model(self):
        from sklearn.externals import joblib
        try:
            self.model = joblib.load(self.filename)
        except IOError:
            pass


class ModelNN(ScikitPredictor):
    '''Nearest neighbor classifier'''

    def generate_model(self, x, y):
        from sklearn.neighbors import KNeighborsRegressor

        self.model = KNeighborsRegressor(**self.model_kwargs)
        self.model.fit(x, y)


class ModelRandomForest(ScikitPredictor):
    '''Random forest classifier'''

    def generate_model(self, x, y):
        from sklearn import ensemble

        # TODO: Try also regressor?

        self.model = ensemble.RandomForestClassifier(**self.model_kwargs)
        self.model.fit(x, y)


def init_models():
    """
    Initialize models and return them as list

    :return: list of PredictionModel
    """
    models = []
    models.append(PredictionModel('0-model', 'data/disruptions_model0.json'))
    models.append(ModelNN('4NN', 'data/4nn.json', 'model/4nn.pkl', n_neighbors=4))
    # models.append(
    #     ModelRandomForest('Optimized Forest', 'data/opt_forest.json', 6, 'model/opt_forest.pkl'))

    return models


def load_models(models):
    """
    Import models from generated pickle files.

    :parameter models: list of PredictionModel
    """
    for model in models:
        model.load_model()


def generate_models(models, xx, yy):
    """
    Generate and save models to pickle files.

    @type  models: list[PredictionModel]
    """
    for model in models:
        model.generate_model(xx, yy)
        model.save_model()
        print('Saved model %s' % model.name)


prediction_models = init_models()

