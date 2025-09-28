import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss cho face recognition
    """

    def __init__(self, embedding_size=128, num_classes=1000, s=64.0, m=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.s = s  # scale factor
        self.m = m  # margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # L2 normalize
        embeddings = F.normalize(embeddings)
        W = F.normalize(self.weight)
        cosine = F.linear(embeddings, W)  # cos(θ)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = cosine * (1 - one_hot) + target * one_hot
        output *= self.s
        loss = F.cross_entropy(output, labels)
        return loss
