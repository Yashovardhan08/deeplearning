import torch 
import torchvision
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self,in_features,heads):
        super(MultiheadAttention, self).__init__()
        self.W_k = nn.Linear(in_features=in_features,out_features=in_features)
        self.W_q = nn.Linear(in_features=in_features,out_features=in_features)
        self.W_v = nn.Linear(in_features=in_features,out_features=in_features)
        self.heads = heads
        
    def forward(self,x):
        batches, t , d = x.shape
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)
        
        # shape before : batches * t * d
        # shape after : batches * t * heads * d/heads
        # shape needed : batches * heads * t * d/heads
        keys = torch.swapaxes( torch.reshape(keys, (batches,t,self.heads,int(d/self.heads))),1,2)
        queries = torch.swapaxes(torch.reshape(queries, (batches,t,self.heads,int(d/self.heads))),1,2)
        values = torch.swapaxes(torch.reshape(values, (batches,t,self.heads,int(d/self.heads))),1,2)
        
        # current shape : batches * heads * t * d/heads
        softmax = nn.Softmax(-1)
        DbyH = (d/self.heads)**(1/2)
        dot = softmax( torch.matmul(keys, torch.swapaxes(queries,-1,-2) ) / DbyH )
        # dot is of the shape: batches * heads * t * t
        dot = torch.reshape(torch.swapaxes(dot @ values,1,2),(batches,t,d))
        return dot
        
        
