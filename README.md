# Дизайн ML системы - классификация новостей по категориям

## Введение

Это приложение использует модель distilBERT для решения задачи мультилейбл классификации новостных статей. Модель обучена на большом корпусе текстов и способна определять несколько категорий, которые соответствуют содержанию новости.

## Цели и предпосылки  

Целью данного приложения является предоставление эффективного инструмента для классификации новостных статей по нескольким категориям, что позволяет улучшить навигацию и поиск по новостным ресурсам. Предпосылками создания приложения являются растущий объем информации в интернете и необходимость её структурирования.  


## Установка модели

Для классификации новостей на русском языке:

```python
# Load model directly
from transformers import DistilBERTClassRus
model = DistilBERTClassRus.from_pretrained("Anzovi/distilBERT-news-ru")
```

Для классификации новостей на английском языке:

```python
# Load model directly
from transformers import DistilBERTClass
model = DistilBERTClass.from_pretrained("Anzovi/distilBERT-news")
```

Или клонирование репозитория для использования с локальной машины:

```python
import classification
```

## Использование

Чтобы классифицировать новостные статьи, выполните следующие шаги (пример для новостей на русском):

1. Импортируйте необходимые модули и загрузите модель:

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader

tokenizer = DistilBertTokenizer.from_pretrained('Anzovi/distilBERT-news')
model = DistilBertForSequenceClassification.from_pretrained('Anzovi/distilBERT-news')
```

2. Подготовьте данные для классификации:

```python
texts = ["Здесь ваш текст новости..."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

3. Классифицируйте тексты:

```python
import torch

with torch.no_grad():
    logits = model(**inputs).logits
```

4. Преобразуйте логиты в вероятности и определите метки:

```python
from torch.nn.functional import softmax

probabilities = softmax(logits, dim=1)
labels = (probabilities > 0.5).nonzero(as_tuple=True)
```

Или локальное использование:
```python
import classification
import pandas as pd

df = pd.read_csv('news.csv')
text_column = 'text'

df_finale = classification.classify(df, text_column)
df_finale.to_csv("Out.csv")
```

## Конфигурация

Модель distilBERT может быть настроена для улучшения результатов классификации. Возможные параметры конфигурации включают количество эпох, размер батча и скорость обучения.

## Лицензия

Это приложение распространяется под лицензией MIT.

---

Документация представлена в кратком виде и может быть расширена в зависимости от потребностей проекта и пользователей.

