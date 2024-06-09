# Importing stock ml libraries
from torch import cuda
import warnings
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import logging
import numpy as np
warnings.simplefilter('ignore')
logging.basicConfig(level=logging.ERROR)

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


def evaluate(df, text_column, lang):

    df = df.reset_index(drop=True)

    if lang == 'ru':
        neural_model = "DmitryPogrebnoy/distilbert-base-russian-cased"
        model_file = 'models/state_dict_pytorch_distilbert_news_rus.bin'
    elif lang == 'eng':
        neural_model = "distilbert-base-cased"
        model_file = 'models/state_dict_pytorch_distilbert_news.bin'

    class DistilBERTClass(torch.nn.Module):
        def __init__(self):
            super(DistilBERTClass, self).__init__()
            self.l1 = DistilBertModel.from_pretrained(neural_model)
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

    def evaluating(testing_loader):
        model.eval()
        fin_outputs = []
        with torch.no_grad():
            for _, data in tqdm(enumerate(testing_loader, 0)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                outputs = model(ids, mask, token_type_ids)
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs

    tokenizer = DistilBertTokenizer.from_pretrained(neural_model, truncation=True,
                                                    do_lower_case=True)

    device = 'cuda' if cuda.is_available() else 'cpu'

    model = DistilBERTClass()
    model.load_state_dict(torch.load(model_file))
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


