import numpy as np
import tensorflow as tf

from sentiment.constant import ModelConstants


def create_dataset(df, text_col, tokenizer, text_pair_col=None, label_col=None) -> tf.data.Dataset:
    input_ids, attention_mask = preprocess_input(df, text_col, tokenizer, text_pair_col)
    if label_col is None:
        dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask, df[label_col]))
    return dataset.map(format_input)


def preprocess_input(df, text_col, tokenizer, text_pair_col=None) -> tuple:
    input_ids = np.zeros((len(df), ModelConstants.SEQ_LEN))
    attention_mask = np.zeros((len(df), ModelConstants.SEQ_LEN))
    df = df.reset_index(drop=True)
    if text_pair_col is None:
        for i, sentence in enumerate(df[text_col]):
            input_ids[i, :], attention_mask[i, :] = _tokenize(sentence, tokenizer, text_pair_col)
    else:
        for i in range(len(df)):
            input_ids[i, :], attention_mask[i, :] = _tokenize(df[text_col][i], tokenizer, df[text_pair_col][i])
    input_ids = tf.cast(input_ids, tf.int32)
    attention_mask = tf.cast(attention_mask, tf.int32)
    return input_ids, attention_mask


def _tokenize(text, tokenizer, text_pair=None) -> tuple:
    tokens = tokenizer.encode_plus(text, text_pair, max_length=ModelConstants.SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']


def format_input(input_ids, masks, labels=None):
    formatted_input = {'input_ids': input_ids, 'attention_mask': masks}
    if labels is not None:
        formatted_input = (formatted_input, labels)
    return formatted_input
