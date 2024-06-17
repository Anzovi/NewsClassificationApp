import streamlit as st
import pandas as pd
import classification

# Title
st.title("Мультиклассовая (multilabel) классификация новостей")

# Sidebar
st.sidebar.subheader("Ручной ввод текста")
agree_manually = st.sidebar.checkbox("Вручную ввести текст для классификации единственной новости")
global df


flag = False

if not agree_manually:

    df = None
    uploaded_data = st.file_uploader(label="Загрузка таблицы", type=['csv', 'xlsx'])

    columnSpecified = st.text_input(
        "Название столбца с текстом новостей таблицы",
        key="fieldSpecified",
    )

    if (uploaded_data is not None):
        try:
            df = pd.read_csv(uploaded_data)
        except Exception as err:
            df = pd.read_excel(uploaded_data)

        if (df is not None) and columnSpecified != '':
            if columnSpecified in df.columns:
                flag = True
            else:
                st.write("Такого столбца нет в таблице")
                flag = False

else:
    textData = st.text_input(
        "Ввод текста одной новости для классификации",
        key="manualData",
    )

    if textData:
        columnSpecified = 'text'
        df = pd.DataFrame(data=[textData], columns=[columnSpecified])
        st.write('Text says: ', textData)
        flag = True
    else:
        flag = False

if flag:
    try:
        # Исходный датафрейм
        st.write(df)

        # Классификация и результат работы
        df_final = classification.classify(df, columnSpecified)
        st.write("Результат обработки", df_final)


        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False)#.encode("utf-8")

        csv = convert_df(df_final)

        st.download_button(
            label="Загрузить в формате CSV",
            data=csv,
            file_name="Output.csv",
            mime="text/csv",
            key='download-csv'
        )
    except Exception as err:
        pass

        #st.write("Добавьте таблицу для классификации")







