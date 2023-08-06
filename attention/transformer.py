import torch
import torchvision
import torch.nn as nn
import attention


class TransformerEncoder(nn.Module):
    def __init__(self,input_features, heads) -> None:
        super(TransformerEncoder,self).__init__()
        self.attentionBlock = attention.MultiheadAttention(in_features=input_features,heads=heads)
        self.norm1 = nn.LayerNorm(input_features)
        self.norm2 = nn.LayerNorm(input_features)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_features,input_features),
            nn.ReLU()
        )
        
    def forward(self,x):
        out = self.attentionBlock(x)
        out = self.norm1(out)
        out = self.feed_forward(out)
        out = self.norm2(out)
        return out