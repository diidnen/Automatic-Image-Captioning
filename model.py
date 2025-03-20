import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

class EnhancedEncoderCNN(nn.Module):
    def __init__(self, embed_size, bn_momentum=0.1, dropout_rate=0.3):
        super().__init__()
        # 主干网络初始化（保持冻结）
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters(): 
            param.requires_grad_(False) 
        
        # 特征提取模块 
        self.feature_extractor  = nn.Sequential(*list(resnet.children())[:-1]) 
        
        # 增强模块 
        self.embed  = nn.Linear(resnet.fc.in_features,  embed_size)
        self.bn  = nn.BatchNorm1d(embed_size, momentum=bn_momentum)
        self.dropout  = nn.Dropout(p=dropout_rate)

    def forward(self, images):
        # 特征提取 [B,3,224,224] → [B,2048,1,1]
        features = self.feature_extractor(images) 
        
        # 维度处理 [B,2048,1,1] → [B,2048]
        features = features.flatten(start_dim=1) 
        
        # 增强处理 
        embedded = self.embed(features)     # [B, embed_size]
        normalized = self.bn(embedded)      # 批量标准化 
        activated = F.relu(normalized)      # 非线性激活 
        features = self.dropout(activated)   # 随机失活 
        
        return features 
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(num_embeddings = vocab_size,
                                  embedding_dim = embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True) 
        
        self.linear = nn.Linear(in_features = hidden_size,
                                out_features = vocab_size)
        
        
        
    
    def forward(self, features, captions):
        captions = captions[:, :-1]#去掉最后一个<end>,captions的维度为[B,L-1]
        embedding = self.embed(captions)#embedding的维度为[B,L-1,E]
        embedding = torch.cat((features.unsqueeze(dim = 1), embedding), dim = 1)#embedding的维度为[B,L,E] features的维度为[B,1,E]
        lstm_out, hidden = self.lstm(embedding)#lstm_out的维度为[B,L,H]
        outputs = self.linear(lstm_out)#outputs的维度为[B,L,V]
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        for index in range(max_len):  #inputs的维度为[B,1,E]
            
            
            lstm_out, states = self.lstm(inputs, states)#lstm_out的维度为[B,1,H]

            
            lstm_out = lstm_out.squeeze(1)#lstm_out的维度为[B,H] 删除index为1的维度
            outputs = self.linear(lstm_out) #outputs的维度为[B,V]
            
            
            target = outputs.max(1)[1]#后面会去cross entropyloss然后不需要softmax
            
            
            predicted_sentence.append(target.item())
            
            
            inputs = self.embed(target).unsqueeze(1)
            
        return predicted_sentence