from email.mime import image
import sys
import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from collections import Counter
import re
from model import DecoderRNN,EnhancedEncoderCNN
import math
import torch.nn as nn


def simple_tokenizer(text):
    # 匹配单词和常见标点
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return [token for token in tokens if token.strip()]


transform_train=transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
])

class CocoCaptionDataset(Dataset):
    def __init__(self, root, annFile, transform=None, mode='train', vocab=None, threshold=5):
        self.root = root
        self.coco = COCO(annFile)
        self.transform = transform
        self.mode = mode
        
        # 获取所有标注ID
        self.ids = list(self.coco.anns.keys())
        
        # 初始化词汇表
        self.vocab = vocab or self.build_vocab(threshold)

    def build_vocab(self, threshold):
        counter = Counter()
        for ann_id in self.ids:
            caption = str(self.coco.anns[ann_id]['caption'])
            tokens = simple_tokenizer(caption)
            counter.update(tokens)
        
        # 创建词汇表
        vocab = Vocabulary()
        vocab.add_word('<pad>')  # 填充标记
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')
        
        # 添加满足频率阈值的单词
        words = [word for word, cnt in counter.items() if cnt >= threshold]
        for word in words:
            vocab.add_word(word)
            
        return vocab

    def __getitem__(self, index):
        try:
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']
            
            full_path = os.path.join(self.root, path)
            if not os.path.exists(full_path):
                print(f"跳过缺失文件: {full_path}")
                return None  # 返回空值
            
            image = Image.open(full_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            tokens = simple_tokenizer(str(caption))
            caption = [self.vocab('<start>')] + [self.vocab(t) for t in tokens] + [self.vocab('<end>')]
            return image, torch.tensor(caption, dtype=torch.long)
        
        except Exception as e:
            print(f"处理索引 {index} 时出错: {str(e)}")
            return None

    def __len__(self):
        return len(self.ids)

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])
        
    def __len__(self):
        # 返回词汇表的大小
        return len(self.word2idx)

def collate_fn(data):
    # 过滤空数据
    valid_data = [d for d in data if d is not None]
    if not valid_data:
        return None, None, None  # 返回空批次
        
    valid_data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*valid_data)
    
    lengths = [len(cap) for cap in captions]
    padded = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap
        
    return torch.stack(images), padded, torch.tensor(lengths)

def get_loader(root, annFile, transform, batch_size=32, shuffle=True, num_workers=2):
    """创建并返回配置好的数据加载器"""
    dataset = CocoCaptionDataset(root=root, 
                                annFile=annFile,
                                transform=transform)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn  # 使用全局函数
    )






