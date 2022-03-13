import json

from xgboost import XGBClassifier


class Model:
    def __init__(self, param_path: str):
        self.param_path = param_path
        self.clf = XGBClassifier

    def get(self):
        model_params = self._get_model_params()
        return self.clf(**model_params)

    def _get_model_params(self) -> dict:
        with open(self.param_path) as f:
            params = json.load(f)
        return params
