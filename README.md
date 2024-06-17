# Документация приложения для мультилейбл классификации новостей с использованием distilBERT

## Введение

Это приложение использует модель distilBERT для решения задачи мультилейбл классификации новостных статей. Модель обучена на большом корпусе текстов и способна определять несколько категорий, которые соответствуют содержанию статьи.

## Установка

Для работы с приложением необходимо установить следующие библиотеки:

```bash
pip install transformers
pip install torch
```

## Использование

Чтобы классифицировать новостные статьи, выполните следующие шаги:

1. Импортируйте необходимые модули и загрузите модель:

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
```

2. Подготовьте данные для классификации:

```python
texts = ["Здесь ваш текст новости..."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

3. Классифицируйте тексты:

```python
with torch.no_grad():
    logits = model(**inputs).logits
```

4. Преобразуйте логиты в вероятности и определите метки:

```python
from torch.nn.functional import softmax

probabilities = softmax(logits, dim=1)
labels = (probabilities > 0.5).nonzero(as_tuple=True)
```

## Конфигурация

Модель distilBERT может быть настроена для улучшения результатов классификации. Возможные параметры конфигурации включают количество эпох, размер батча и скорость обучения.

## Лицензия

Это приложение распространяется под лицензией MIT.

---

Документация представлена в кратком виде и может быть расширена в зависимости от потребностей проекта и пользователей.

