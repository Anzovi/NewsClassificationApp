import pandas as pd
import classification


df = pd.DataFrame(data=[text], columns=['text'])
text_column = 'text'

# Передать в функцию датафрейм и название столбца с текстом новостей
df_finale = classification.classify(df, text_column)
df_finale.to_csv("Out.csv")
