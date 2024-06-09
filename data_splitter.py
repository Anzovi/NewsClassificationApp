import pandas as pd
from langdetect import detect


def datasplitter(df, text_column):
    df[text_column] = df[text_column].astype(str)
    df = df[df[text_column].notna()]
    df_rus = df[df.apply(lambda x: detect(x[text_column]) == 'ru', axis=1)]
    df_eng = df[df.apply(lambda x: detect(x[text_column]) == 'en', axis=1)]
    return df_eng, df_rus


