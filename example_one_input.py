import pandas as pd
import classification

text = input("Введите текст новости: ")
text_column = 'text'

df = pd.DataFrame(data=[text], columns=[text_column])

# Передать в функцию датафрейм и название столбца с текстом новостей
df_finale = classification.classify(df, text_column)
df_finale.to_csv("Out.csv")

