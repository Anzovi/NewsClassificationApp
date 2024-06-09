import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pymorphy3

nltk.download('stopwords')
nltk.download('punkt')
set_stopwords = set(stopwords.words('russian'))
set_punctuation = string.punctuation
set_punctuation = set_punctuation + '«' + '»' + '``' + "''" + "/" + "-"


def preprocessing_df_rus(df, text_column):
    def preprocess(text, stop_words, punctuation_marks, morph):
        tokens = word_tokenize(text.lower())
        preprocessed_text = []
        for token in tokens:
            if token not in punctuation_marks:
                lemma = morph.parse(token)[0].normal_form
                if lemma not in stop_words:
                    preprocessed_text.append(lemma)
        return preprocessed_text



    morph = pymorphy3.MorphAnalyzer()
    df[text_column] = pd.Series(df[text_column]).apply(
        lambda text: preprocess(text, set_stopwords, set_punctuation, morph))

    df[text_column] = pd.Series(df[text_column]).apply(lambda x: ' '.join(x))

    return df
