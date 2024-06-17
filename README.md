# Дизайн ML системы - классификация новостей по категориям

## Введение

Это приложение использует модель distilBERT для решения задачи мультилейбл классификации новостных статей. Модель обучена на большом корпусе текстов и способна определять несколько категорий, которые соответствуют содержанию новости.  

Пример результата классификации из пайплайна для одной новости:  
|Внешняя торговля | Инвестиции | Пункты пропуска | Санкции | Совместные проекты и программы | Специальные отношения, не классифицированные по типу |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
| 1 | 0 | 1 | 1 | 0 | 0 |  

Для удобства использования модели пользователем написано приложение на streamlit: main.py

## Цели и предпосылки  

Целью данного приложения является предоставление эффективного инструмента для классификации новостных статей по нескольким категориям, что позволяет улучшить навигацию и поиск по новостным ресурсам. Предпосылками создания приложения являются растущий объем информации в интернете и необходимость её структурирования.  

## Методология

Методология проекта включает в себя использование предварительно обученной модели distilBERT, которая была адаптирована для задачи мультилейбл классификации. Модель обучается на выборке новостных статей с уже установленными категориями.  

Блок-схема с ключевыми этапами решения задачи:  
<p align="center">
  <img src="https://github.com/Anzovi/NewsClassificationApp/blob/main/schema.png"/>
</p>


## Установка модели

Ссылка на модель для классификации новостей на русском: https://huggingface.co/Anzovi/distilBERT-news-ru  
Ссылка на модель для классификации новостей на английском: https://huggingface.co/Anzovi/distilBERT-news  

Для классификации новостей только на русском языке:

```python
# Load model directly
from transformers import DistilBERTClassRus
model = DistilBERTClassRus.from_pretrained("Anzovi/distilBERT-news-ru")
```

Для классификации новостей только на английском языке:

```python
# Load model directly
from transformers import DistilBERTClass
model = DistilBERTClass.from_pretrained("Anzovi/distilBERT-news")
```

Или клонирование репозитория для использования с пайплайном (и для английского и для русского):

```python
import classification
```

## Использование

### Чтобы классифицировать новостные статьи есть два способа:  

### Напрямую через обращение к модели  

1. Импортируйте необходимые модули и загрузите модель:

```python
from distilBERTModule import DistilBERTClassRus # Локально
# from transformers import DistilBERTClass # Из облака
from transformers import AutoTokenizer

device = "cuda"

# Локально, если папка с моделью названа - models_rus
tokenizer = AutoTokenizer.from_pretrained('models_rus')
model = DistilBERTClassRus.from_pretrained('models_rus')

# Или из облака
tokenizer = AutoTokenizer.from_pretrained('Anzovi/distilBERT-news-ru')
model = DistilBERTClassRus.from_pretrained('Anzovi/distilBERT-news-ru')

model.to(device)
```

2. Подготовьте данные для классификации:

```python
text = "Здесь ваш текст новости..."
encoded_input = tokenizer(text, return_tensors='pt', return_token_type_ids=True).to(device)
```

3. Классифицируйте тексты:

```python
import torch

with torch.no_grad():
    outputs = model(**encoded_input)
```

4. Преобразуйте вероятностные значения классов в категории:

```python
fin_outputs = []
fin_outputs.extend(((torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5).astype(int)).tolist())

classes_names = ['Внешняя торговля',
                   'Инвестиции',
                   'Пункты пропуска',
                   'Санкции',
                   'Совместные проекты и программы',
                   'Специальные отношения, не классифицированные по типу']

import pandas as pd

labels = pd.DataFrame(data=fin_outputs, columns=classes_names)

print(labels)
```

### Использование заготовленного пайплаина
Для *.csv файла:
```python
import classification
import pandas as pd

df = pd.read_csv('news.csv')
text_column = 'text'

df_finale = classification.classify(df, text_column)
df_finale.to_csv("Out.csv")
```

Для одного текста:
```python
import classification
import pandas as pd

text = "Здесь ваш текст новости..."
df = pd.DataFrame(data=[text], columns=['text'])
text_column = 'text'

df_finale = classification.classify(df, text_column)
print(df_finale)
```

## Издержки и Риски

Издержки включают затраты на обучение модели, серверное оборудование и поддержку. Основные риски связаны с возможностью неправильной классификации и необходимостью постоянного обновления модели.


## Конфигурация

Модель distilBERT может быть настроена для улучшения результатов классификации. Возможные параметры конфигурации включают количество эпох, размер батча и скорость обучения.

## Качество классификации модели
Где 0 - Внешняя торговля, 1 - Инвестиции, 2 - Пункты пропуска, 3 - Санкции, 4 - Совместные проекты и программы, 5 - Специальные отношения, не классифицированные по типу  

Для модели новостей на русском языке:  

              precision    recall  f1-score   support

           0       0.91      0.88      0.90      1644
           1       0.85      0.71      0.77       270
           2       0.35      0.67      0.46        33
           3       0.77      0.73      0.75       329
           4       0.73      0.65      0.69       464
           5       0.66      0.65      0.65       437
   micro avg       0.82      0.78      0.80      3177
   macro avg       0.71      0.71      0.70      3177
weighted avg       0.83      0.78      0.80      3177
 samples avg       0.86      0.84      0.83      3177


Для модели новостей на английском языке:  

              precision    recall  f1-score   support

           0       0.85      0.89      0.87       659
           1       0.76      0.84      0.80       235
           2       0.35      0.47      0.40        15
           3       0.74      0.52      0.61        48
           4       0.75      0.73      0.74       313
           5       0.69      0.69      0.69       266
   micro avg       0.78      0.80      0.79      1536
   macro avg       0.69      0.69      0.68      1536
weighted avg       0.78      0.80      0.79      1536
 samples avg       0.83      0.84      0.81      1536
 

## Ожидаемые выгоды

Ожидаемые выгоды от внедрения приложения включают экономию времени работника новостного отделения на категоризацию новостей.

