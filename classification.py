import pandas as pd
import warnings
import logging

import data_splitter
import data_processing_rus
import data_processing_eng
import distilBERTModule

warnings.simplefilter('ignore')
logging.basicConfig(level=logging.ERROR)

'''
Модуль классификации новостей - главный
Входные: дата фрейм с исходными новостями
Выходные: сохранение csv файла Out.csv с исходными данными на всех языках, полем index для соответствия новых новостей 
с старыми, бинарно-квалифицированными заполненными полями новостей и полем принадлежности новости 
к русскому или английскому языку.
'''


def classify(df, columnSpecified):

    if (df is not None) and (columnSpecified != ''):

        if columnSpecified in df.columns:
            # Исходный датафрейм
            df = df.reset_index()

            # Разделение по языкам
            df_eng, df_rus = data_splitter.datasplitter(df, columnSpecified)

            # Предобработка текста и классификация
            df_rus_classified = pd.DataFrame(columns=df.columns)
            df_eng_classified = pd.DataFrame(columns=df.columns)

            if df_rus.shape[0] > 0:
                df_rus_processed = data_processing_rus.preprocessing_df_rus(df_rus, columnSpecified)
                df_rus_classified = distilBERTModule.evaluate(df_rus_processed, columnSpecified, 'ru')
            if df_eng.shape[0] > 0:
                df_eng_processed = data_processing_eng.preprocessing_df_eng(df_eng, columnSpecified)
                df_eng_classified = distilBERTModule.evaluate(df_eng_processed, columnSpecified, 'eng')

            # Объединение новостей на разных языках

            df_final = pd.concat([df_rus_classified, df_eng_classified]).reset_index(drop=True)

            return df_final
            #df_final.to_csv('Out.csv')
        else:
            print("Такого столбца нет в таблице")
    else:
        print('Введите корректные данные')


