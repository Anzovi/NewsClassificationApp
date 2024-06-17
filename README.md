# Дизайн ML системы - классификация новостей по категориям

## Введение

Это приложение использует модель distilBERT для решения задачи мультилейбл классификации новостных статей. Модель обучена на большом корпусе текстов и способна определять несколько категорий, которые соответствуют содержанию новости.  

Пример результата классификации из пайплайна для одной новости:  
|Внешняя торговля | Инвестиции | Пункты пропуска | Санкции | Совместные проекты и программы | Специальные отношения, не классифицированные по типу |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
| 1 | 0 | 1 | 1 | 0 | 0 |  

## Цели и предпосылки  

Целью данного приложения является предоставление эффективного инструмента для классификации новостных статей по нескольким категориям, что позволяет улучшить навигацию и поиск по новостным ресурсам. Предпосылками создания приложения являются растущий объем информации в интернете и необходимость её структурирования.  


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

# Локально
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

## Конфигурация

Модель distilBERT может быть настроена для улучшения результатов классификации. Возможные параметры конфигурации включают количество эпох, размер батча и скорость обучения.

