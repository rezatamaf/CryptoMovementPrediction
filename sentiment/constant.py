class ModelConstants:
    SEQ_LEN = 128
    BERT_FLAVOR = "vinai/bertweet-base"
    ASPECT_CATEGORY = ['ECONOMY', 'TECHNOLOGY', 'GENERAL', 'COMMUNITY', 'SECURITY']
    ID2POLARITY = {0: 'POSITIVE', 1: 'NEGATIVE', 2: 'NONE'}


class NLPDataConstants:
    COL_NAMES = {'DATE': 'date', 'TITLE': 'title', 'ARTICLE': 'article', 'MEDIA': 'media'}
    COL_DTYPES = {'article': str, 'title': str, 'media': str}

