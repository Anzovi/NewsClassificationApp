# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import resnet50, ResNet50_Weights
import streamlit as st
import pandas as pd


import data_splitter
import Data_processing_rus
import Data_processing_eng
import NNModel


# Title
st.title("Мультиклассовая (multilabel) классификация новостей")

# Sidebar
st.sidebar.subheader("Ручной ввод текста")
agree_manually = st.sidebar.checkbox("Вручную ввести текст для классификации единственной новости")


if not agree_manually:
    global df
    uploaded_data = st.file_uploader(label="Загрузка таблицы", type=['csv', 'xlsx'])

    columnSpecified = st.text_input(
        "Название столбца с текстом новостей таблицы",
        key="fieldSpecified",
    )

    if (uploaded_data is not None) and (columnSpecified != ''):
        try:
            df = pd.read_csv(uploaded_data)
        except Exception as err:
            df = pd.read_excel(uploaded_data)

    try:
        if columnSpecified in df.columns:
            # Исходный датафрейм
            df = df.reset_index()
            st.write(df)

            # Разделение по языкам
            df_eng, df_rus = data_splitter.datasplitter(df, columnSpecified)

            # Предобработка текста
            df_rus_processed = Data_processing_rus.preprocessing_df_rus(df_rus, columnSpecified)
            df_eng_processed = Data_processing_eng.preprocessing_df_eng(df_eng, columnSpecified)

            # Классификация
            df_rus_classified = NNModel.evaluate(df_rus_processed, columnSpecified, 'ru')
            df_eng_classified = NNModel.evaluate(df_eng_processed, columnSpecified, 'eng')

            # Объединение новостей на разных языках
            df_final = pd.concat([df_rus_classified, df_eng_classified]).reset_index(drop=True)
            st.write("Результат обработки", df_final)


            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv(index=False).encode("utf-8")

            csv = convert_df(df_final)

            st.download_button(
                label="Загрузить в формате CSV",
                data=csv,
                file_name="Output.csv",
                mime="text/csv",
                key='download-csv'
            )

        else:
            st.write("Такого столбца нет в таблице")
    except Exception as err:
        st.write("Добавьте таблицу для классификации")

else:
    textData = st.text_input(
        "Ввод текста одной новости для классификации",
        key="manualData",
    )
    if textData:
        st.write('Text says: ', textData)





