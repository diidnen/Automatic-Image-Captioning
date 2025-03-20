import numpy as np
import os
import requests
import time
import sys
import torch
import torch.nn as nn
import torch.utils.data as data
import math
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def train_val_split(dataset, val_split=0.2, shuffle=True, random_seed=42):
    """将数据集分割为训练集和验证集
    
    参数:
        dataset: 完整数据集
        val_split: 验证集比例 (0-1)
        shuffle: 是否打乱数据
        random_seed: 随机种子
        
    返回:
        train_sampler, val_sampler: 用于DataLoader的采样器
    """
    # 获取数据集大小
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    # 随机打乱顺序
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    # 分割为训练和验证索引
    train_indices, val_indices = indices[split:], indices[:split]
    
    # 创建采样器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    return train_sampler, val_sampler

def get_split_data_loaders(root, annFile, transform, batch_size=32, num_workers=2, val_split=0.2):
    """创建训练和验证数据加载器
    
    参数:
        root: 图像目录
        annFile: 标注文件路径
        transform: 图像转换
        batch_size: 批次大小
        num_workers: 工作线程数
        val_split: 验证集比例
        
    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    from Dataloader import CocoCaptionDataset, collate_fn
    
    # 创建完整数据集
    dataset = CocoCaptionDataset(root=root, annFile=annFile, transform=transform)
    
    # 分割为训练集和验证集
    train_sampler, val_sampler = train_val_split(dataset, val_split=val_split)
    
    # 创建数据加载器
    train_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def validate(val_loader, encoder, decoder, criterion, vocab_size, device):
    """验证模型性能
    
    参数:
        val_loader: 验证数据加载器
        encoder: CNN编码器
        decoder: RNN解码器
        criterion: 损失函数
        vocab_size: 词汇表大小
        device: 计算设备
        
    返回:
        float: 平均验证损失
    """
    # 设置为评估模式
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for i, (images, captions, _) in enumerate(val_loader):
            # 跳过空批次
            if images is None:
                continue
                
            # 移至设备
            images = images.to(device)
            captions = captions.to(device)
            
            # 前向传播
            features = encoder(images)
            outputs = decoder(features, captions)
            
            # 计算损失
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            # 累加损失
            total_loss += loss.item()
            valid_batches += 1
    
    # 计算平均损失
    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
    
    # 恢复训练模式
    encoder.train()
    decoder.train()
    
    return avg_loss

def train(train_loader, val_loader, encoder, decoder, num_epochs, log_file='training_log.txt', 
          print_every=100, save_every=1, vocab_size=None):
    """训练图像描述生成模型，包含验证步骤
    
    参数:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        encoder: CNN编码器模型
        decoder: RNN解码器模型
        num_epochs: 训练轮数
        log_file: 日志文件路径
        print_every: 打印频率
        save_every: 保存模型频率
        vocab_size: 词汇表大小
    """
    # 创建模型保存目录
    os.makedirs('./models', exist_ok=True)
    
    # 确保词汇表大小已定义
    if vocab_size is None:
        vocab_size = len(train_loader.dataset.vocab)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = torch.optim.Adam(params=params, lr=0.001)
    
    # 计算每轮训练的总步数
    total_step = len(train_loader)
    
    # 打开日志文件
    f = open(log_file, 'w')
    
    # Google Colab保活设置
    old_time = time.time()
    try:
        response = requests.request("GET", 
                                "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token", 
                                headers={"Metadata-Flavor":"Google"})
    except:
        print("不在Google Colab环境中运行")
        response = None
    
    # 跟踪最佳验证损失
    best_val_loss = float('inf')
    
    # 开始训练循环
    for epoch in range(1, num_epochs+1):
        # 训练阶段
        for i_step, (images, captions, _) in enumerate(train_loader, 1):
            # 跳过空批次
            if images is None:
                continue
                
            # Google Colab保活请求
            if response and time.time() - old_time > 60:
                old_time = time.time()
                requests.request("POST", 
                                "https://nebula.udacity.com/api/v1/remote/keep-alive", 
                                headers={'Authorization': "STAR " + response.text})
            
            # 将图像和标题批次移至GPU（如果可用）
            images = images.to(device)
            captions = captions.to(device)
            
            # 梯度清零
            decoder.zero_grad()
            encoder.zero_grad()
            
            # 通过CNN-RNN模型传递输入
            features = encoder(images)
            outputs = decoder(features, captions)
            
            # 计算批次损失
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            # 反向传播
            loss.backward()
            
            # 更新优化器中的参数
            optimizer.step()
                
            # 获取训练统计数据
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
            
            # 打印训练统计数据（同一行）
            print('\r' + stats, end="")
            sys.stdout.flush()
            
            # 将训练统计数据写入文件
            f.write(stats + '\n')
            f.flush()
            
            # 打印训练统计数据（不同行）
            if i_step % print_every == 0:
                print('\r' + stats)
        
        # 验证阶段
        val_loss = validate(val_loader, encoder, decoder, criterion, vocab_size, device)
        val_stats = 'Epoch [%d/%d], Validation Loss: %.4f, Validation Perplexity: %5.4f' % (epoch, num_epochs, val_loss, np.exp(val_loss))
        print(val_stats)
        f.write(val_stats + '\n')
        f.flush()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-best.pkl'))
            torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-best.pkl'))
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
                
        # 定期保存权重
        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), os.path.join('./models', f'decoder-{epoch}.pkl'))
            torch.save(encoder.state_dict(), os.path.join('./models', f'encoder-{epoch}.pkl'))
    
    # 关闭训练日志文件
    f.close()
    print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")

# 示例用法
if __name__ == "__main__":
    from model import EnhancedEncoderCNN, DecoderRNN
    
    # 设置参数
    batch_size = 32
    embed_size = 256
    hidden_size = 512
    num_epochs = 5
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # 创建训练和验证数据加载器
    train_loader, val_loader = get_split_data_loaders(
        root='data/coco/train2014',
        annFile='data/coco/annotations/captions_train2014.json',
        transform=transform,
        batch_size=batch_size,
        val_split=0.2  # 20%验证，80%训练
    )
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    vocab_size = len(train_loader.dataset.vocab)
    encoder = EnhancedEncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
    
    # 开始训练
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        encoder=encoder,
        decoder=decoder,
        num_epochs=num_epochs,
        vocab_size=vocab_size
    )
