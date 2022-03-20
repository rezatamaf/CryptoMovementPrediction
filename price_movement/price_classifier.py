import json

from xgboost import XGBClassifier


class Model:
    def __init__(self, param_path: str = None, params: dict = None):
        self.param_path = param_path
        self.params = params
        self.clf = XGBClassifier

    def get(self):
        return self.load_param(self.param_path, self.params)

    def load_param(self, param_path: str = None, params: dict = None):
        if param_path is not None:
            model_params = self._get_model_params()
            clf = self.clf(**model_params)
        elif params is not None:
            clf = self.clf(**self.params)
        else:
            clf = self.clf()
        return clf

    def _get_model_params(self) -> dict:
        with open(self.param_path) as f:
            params = json.load(f)
        return params
