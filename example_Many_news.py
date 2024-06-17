import pandas as pd
import classification

df = pd.read_csv('new_test_small.csv')
text_column = 'text'

# Передать в функцию датафрейм и название столбца с текстом новостей
df_finale = classification.classify(df, text_column)
df_finale.to_csv("Out.csv")
