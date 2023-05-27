import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        
        self.in_features = in_features
        activation = nn.Softmax(dim=1)
        
        self.attention_pooling = nn.Sequential(
            nn.Conv1d(
                in_channels=in_features,
                out_channels=1,
                kernel_size=1,
            ),
            activation
        )
        
    def forward(
        self,
        x
            ):
        batch_size, history_len, _ = x.shape
        
        x = x.view(batch_size, history_len, -1)
        x_a = x.transpose(1, 2)
        x_attn = (self.attention_pooling(x_a) * x_a).transpose(1, 2)
        x_attn = x_attn.sum(1, keepdim=True)
        
        return x_attn


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        x
            ):
        return torch.mean(x, dim=1)
    

class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        x
            ):
        return torch.max(x, dim=1).values


class MinPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        x
            ):
        return torch.min(x, dim=1).values
