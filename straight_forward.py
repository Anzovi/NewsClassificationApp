import torch
import pandas as pd
from distilBERTModule import DistilBERTClassRus
from transformers import AutoTokenizer

# Напрямую

device = "cuda"

text = u"ожидаться финансирование привлечь течение следующий четыре год рио-де-жанейро 27 март /тасс/ президент бразилия луис инасиу лул силва находиться визит южноамериканский страна президент франция эмманюэль макрон подписать вторник соглашение партнёрство который предполагать привлечение инвестиция размер €1 млрд проект охрана развитие бразильский амазонии французский гвиана это сообщать местный новостной портал metropoles отмечаться финансирование предполагаться привлечь течение близкий четыре год партнёр проект стать ведущий банк бразилия французский агентство развитие соглашение также предусматривать развитие исследовательский проект связанный устойчивый развитие бразилия французский гвиана макрон находиться бразилия официальный визит 26 28 март качество основный обсуждение заявить сотрудничество лицо основный глобальный проблема частность изменение климат экономический отношение"

tokenizer = AutoTokenizer.from_pretrained('models_rus')
model = DistilBERTClassRus.from_pretrained('models_rus')

encoded_input = tokenizer(text, return_tensors='pt', return_token_type_ids=True).to(device)
model.to(device)

with torch.no_grad():
    outputs = model(**encoded_input)


fin_outputs = []
fin_outputs.extend(((torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5).astype(int)).tolist())

classes_names = ['Внешняя торговля',
                   'Инвестиции',
                   'Пункты пропуска',
                   'Санкции',
                   'Совместные проекты и программы',
                   'Специальные отношения, не классифицированные по типу']

labels = pd.DataFrame(data=fin_outputs, columns=classes_names)

print(labels)
