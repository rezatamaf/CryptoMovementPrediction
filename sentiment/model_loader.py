import transformers

from sentiment.language_model import LstmBertweet
from sentiment.constant import ModelConstants


class ModelLoader:
    def __init__(self, relevance_model_path: str, tabsa_model_path: str):
        self.relevance_model_path = relevance_model_path
        self.tabsa_model_path = tabsa_model_path
        self.tokenizer = self._load_tokenizer()
        self.relevance_model = LstmBertweet(2).build_graph(ModelConstants.SEQ_LEN)
        self.tabsa_model = LstmBertweet(3).build_graph(ModelConstants.SEQ_LEN)

    def load_weights(self):
        self._load_relevance_model_weight()
        self._load_tabsa_model_weight()

    @staticmethod
    def _load_tokenizer():
        tokenizer = transformers.AutoTokenizer.from_pretrained(ModelConstants.BERT_FLAVOR, use_fast=False)
        return tokenizer

    def _load_relevance_model_weight(self):
        self.relevance_model.load_weights(self.relevance_model_path)

    def _load_tabsa_model_weight(self):
        self.tabsa_model.load_weights(self.tabsa_model_path)
