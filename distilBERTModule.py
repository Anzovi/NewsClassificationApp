import torch
import pandas as pd
import numpy as np
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers import DistilBertModel
from torch.utils.data import Dataset
from torch import cuda
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

'''
Модуль для импорта классов моделей и датасетов
Функция evaluate
Входные: дата фрейм с новостями на одном языке с нормализованными словами
Выходные: дата фрейм с полем index для соответствия новых новостей с старыми, бинарно-квалифицированными
заполненными полями новостей и полем принадлежности новости к русскому или английскому языку.
'''

device = 'cuda' if cuda.is_available() else 'cpu'
MAX_LEN = 128


class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, text_column, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe[text_column]
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }


class DistilBERTClassRusConfig(PretrainedConfig):
    model_type = "DistilBERTClassRus"


class DistilBERTClassRus(PreTrainedModel):
    config_class = DistilBERTClassRusConfig

    def __init__(self, config):
        super(DistilBERTClassRus, self).__init__(config)
        self.l1 = DistilBertModel.from_pretrained("DmitryPogrebnoy/distilbert-base-russian-cased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


class DistilBERTClassConfig(PretrainedConfig):
    model_type="DistilBERTClass"


class DistilBERTClass(PreTrainedModel):
    config_class = DistilBERTClassConfig

    def __init__(self, config):
        super(DistilBERTClass, self).__init__(config)
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def evaluate(df, text_column, lang):

    df = df.reset_index(drop=True)

    def evaluating(testing_loader):
        model.eval()
        fin_outputs = []
        with torch.no_grad():
            for data in testing_loader:
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                outputs= model(ids, mask, token_type_ids)
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs

    if lang == 'ru':
        neural_model = "models_rus"
        model = DistilBERTClassRus.from_pretrained(neural_model, local_files_only=True, use_safetensors=True)
    else:
        neural_model = "models"
        model = DistilBERTClass.from_pretrained(neural_model, local_files_only=True, use_safetensors=True)

    tokenizer = AutoTokenizer.from_pretrained(neural_model, local_files_only=True, truncation=True, do_lower_case=True)
    model.to(device)

    testing_set = MultiLabelDataset(df, text_column, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set)

    outputs = evaluating(testing_loader)
    final_outputs = np.array(outputs) >= 0.5

    classes_names = pd.Series(['Внешняя торговля',
                               'Инвестиции',
                               'Пункты пропуска',
                               'Санкции',
                               'Совместные проекты и программы',
                               'Специальные отношения, не классифицированные по типу',
                               'Не по теме – вспомогательный класс'])

    aux = pd.DataFrame(np.hstack(
        (final_outputs.astype(int), np.array([(~np.any(final_outputs, axis=1)).astype(int)]).T)),
                       columns=classes_names)

    df = pd.concat([df, aux], axis=1)
    df['Language'] = lang
    return df
