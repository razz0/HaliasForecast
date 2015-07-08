"""Prediction models for Halias migration prediction based on weather forecast"""

from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import ensemble

import numpy as np

class PredictionModel(object):
    """
    Abstract model class
    """

    def __init__(self, name, json_file, **model_kwargs):
        self.name = name
        self.JSON_FILE = json_file
        self.model_kwargs = model_kwargs
        self.predictions = {}
        self.stored_predictions = {}
        self.model = None  # Prediction model

    def predict(self, *args):
        return 0

    def generate_model(self):
        pass

    def fit_model(self, x, y):
        pass

    def save_model(self):
        pass


class Model0(PredictionModel):
    '''Baseline model (always 0)'''

    def predict(self, *args):
        return 0


class ScikitPredictor(PredictionModel):
    """Pre-calculated Scikit-learn prediction model"""

    def __init__(self, name, json_file, model_file, **model_kwargs):
        super(ScikitPredictor, self).__init__(name, json_file, **model_kwargs)
        self.model = None
        self.filename = model_file

    def predict(self, *args):
        val = self.model.predict(*args)
        if type(val) in [list, np.ndarray]:
            if len(val) > 1:
                raise Exception('Predicted multiple values')
            return int(round(val[0]))
        else:
            return int(round(val))

    def save_model(self):
        joblib.dump(self, self.filename)


class ModelNN(ScikitPredictor):
    '''Nearest neighbor classifier'''

    def generate_model(self):
        self.model = KNeighborsClassifier(**self.model_kwargs)

    def fit_model(self, x, y):
        self.model.fit(x, y)



class ModelNNReg(ScikitPredictor):
    '''Nearest neighbor regression'''

    def generate_model(self):
        self.model = KNeighborsRegressor(**self.model_kwargs)

    def fit_model(self, x, y):
        self.model.fit(x, y)


class ModelRandomForest(ScikitPredictor):
    '''Random forest classifier'''

    def generate_model(self):

        # TODO: Try also regressor?

        self.model = ensemble.RandomForestClassifier(**self.model_kwargs)

    def fit_model(self, x, y):
        self.model.fit(x, y)


def init_models():
    """
    Initialize models and return them as list

    :return: list of PredictionModel
    """
    models = []
    models.append(PredictionModel('0-model', 'data/0model.json'))
    models.append(ModelNN('4NN', 'data/4nn.json', 'model/4nn.pkl', n_neighbors=4))
    # models.append(ModelNNReg('4NN regressor', 'data/4nnr.json', 'model/4nnr.pkl', n_neighbors=4))
    # models.append(ModelNNReg('4NN regressor', 'data/4nnr.json', 'model/4nnr.pkl', n_neighbors=4, algorithm='auto'))
    # models.append(
    #     ModelRandomForest('Optimized Forest', 'data/opt_forest.json', 6, 'model/opt_forest.pkl'))

    return models


def load_models(models, filenames):
    """
    Import models from generated pickle files.

    :parameter models: list of PredictionModel
    """
    models = []
    for filename in filenames:
        model = joblib.load(filename)
        models.append(model)


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

