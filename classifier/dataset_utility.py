import csv
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import json
import torch
import torch.nn.functional as F

    
class NewsDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer =AutoTokenizer.from_pretrained('microsoft/deberta-v3-base',use_fast=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        query = [x[0] for x in data]
        labels = [int(x[1]) for x in data]

        queryEncdoing = self.tokenizer(query, return_tensors='pt',padding=True, truncation=True, max_length=512)

        token_ids = torch.LongTensor(queryEncdoing['input_ids'])
        attention_mask = torch.LongTensor(queryEncdoing['attention_mask'])
        labels = torch.LongTensor(labels)

        return (token_ids, attention_mask,
                labels)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,
         labels) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'labels': labels,
            }

        return batched_data
    

def load_news_data(file_name, split='train'):
    news_data = []
    labels = []
    if split == 'test':
        with open(file_name, 'r') as fp:
            for line in fp:
                line = json.loads(line)
                url = list(line.keys())[0]
                news_data.append((url,line[url]['article']))
    else:
        with open(file_name, 'r') as fp:
            ind = 0 
            for line in fp:
                line = json.loads(line)
                news_data.append(line['article'],
                                        line['label'])
              
            print(len(toxic_data))

    return news_data, {'0','1'}


