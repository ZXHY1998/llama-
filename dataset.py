import torch
import json
from dataclasses import dataclass
from datasets import load_from_disk
from main import log_dist

class JsonlDatasetPT(torch.utils.data.Dataset):
    """
        用于加载json格式的数据集，用于预训练任务
    """

    def __init__(self,
                 data_path,   # 数据集路径
                 tokenizer,   # 分词器实例
                 max_length,  # 最大长度
                 ):
        # 加载数据集并进行词令化
        self.dataset =[]
        with open(data_path,'r',encoding= 'utf-8')as f:
            for line in f:
                text=json.loads(line)['text']
                # 使用tokenizer对旬子进行词令化
                inputs=tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max length',
                    return_tensors='pt',
                    truncation=True
                )
                input_ids = inputs['input_ids'].squeeze() # shape :[max_length]

                # 将词令化后的样本添加到dataset中
                self.dataset.append({
                    'input_ids':input_ids,
                })

        log_dist(f'Loaded {len(self.dataset)} samples from {data_path}')

    def __len__(self):
        # 返回数据集大小
        return len(self.dataset)

    def __getitem__(self, idx):
        # 返回一个样本
        return self.dataset[idx]

def get_pt_dataset(args):
    """
        加载已经词令化的数据集，用于预训练任务
    """
    # 从磁盘加载数据集， 注意该数据集必须是通过save_to_disk()函数保存的
    train_dataset = load_from_disk(args.data_path)
    train_dataset = train_dataset.shuffle(seed=42)
    return train_dataset

class JsonDatasetSFT(torch.utils.data.Dataset):
    """
        加载json格式的数据集，用于指令微调任务
    """
    def __init__(self,
                 data_path,   # 数据集路径
                 tokenizer,   # 分词器实例
                 max_length,  # 最大长度
                 ):
        super().__init__()

        self.dataset = []
        with open(data_path, 'r') as file:
            for line in file:
                sample = json.loads(line)

                sentence = sample['instruction'] + sample['response']
                # 使用tokenizer对句子进行词令化
                tokenized = tokenizer(
                    sentence,
                    max_length = max_length,
                    padding = "max_length",
                    truncation = True,
                    return_tensors="pt")
                tokenized["input_ids"] = tokenized["input _ids"].squeeze(0)
                tokenized["attention mask"] = tokenized["attention mask"].squeeze(0)
                # 将词令化后的样本添加到dataset变量中
                self.dataset.append(tokenized)

            log_dist(f'Loaded {len(self.dataset)} examples from {data_path}')

    def __len__(self):
        # 返回数据集大小
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):

        # 返回一个样本
        return {
            "input_ids": self.dataset[idx]["input_ids"],
            "labels": self.dataset[idx]["input_ids"],
            "attention_mask": self.dataset[idx]["attention_mask"]
        }

@dataclass
class DataCollatorForPT(object):
    """
        Data collator函数，将多个样本拼接成一个batch，同时生成labels，用于计算
        lose。该函数用于预训练任务
    """

    pad_token_id: int = 0
    ignore_index: int = -100
    max_length: int=-1 # 默认不进行max_length截断
    def __ca11__(self,instances:list)-> dict:

        if self.max_length > 0:
            input_ids = torch.stack([instance['input_ids'][:self.max_length] for
                                     instance in instances],dim=0) # shape:[batch_size, max length]
        else:
            input_ids = torch.stack([instance['input ids'] for instance in
                instances],dim=0) # shape: [batch_size,max_length]
        labels=input_ids.clone()
        # 将labels中的pad部分置为ignore_index，计算loss时要忽略
        labels[labels == self.pad_token_id]= self.ignore_index
        return dict(
            input_ids=input_ids,
            labels=labels,
        )
@dataclass
class DataCollatorForSFT(object):
    """
        Data collator函数，将多个样本拼接成一个batch，同时生成labels和
            attention_mask:用于计算loss。该函数用于指令微调任务
    """
    pad_token_id: int = 0

    def __call__(self,fratures):
        len_ids = [len(feature['input_ids']) for feature in fratures] # [14,6,7,10,...]
        longest = max(len_ids) # 14
        input_ids_list = []
        labels_list = []

        # 从长到短排列
        for ids_l,feature in sorted(zip(len_ids,fratures),key= lambda x:-x[0]):

            ids = feature['input_ids']
            labels = feature['labels']
            labels = labels[:len(ids)] # 截断

            # padding补齐
            ids += [self.pad_token_id] * (longest - ids_l)
            # padding部分设置为-100， 使得计算loss时对应值为一个很小的负数， 达到忽略的效果
            labels += [-100] * (longest - ids_l)

            input_ids_list.append(torch.LongTensor(ids))
            labels_list.append(torch.LongTensor(labels))

        input_ids = torch.strackt(input_ids_list) # shape:[batch_size, longest]
        labels = torch.stack(labels_list) # shape:[batch_size, longest]
        return {
            'input_ids':input_ids,
            'labels':labels,
            'attention_mask': input_ids.ne(self.pad_token_id)
        }

