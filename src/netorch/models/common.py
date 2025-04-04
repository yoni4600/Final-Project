#coding:utf-8
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim

class SkipGramNS(nn.Module):
    def __init__(self, num_nodes, dimension, device='cuda:0'):
        super().__init__()
        self.num_nodes = num_nodes
        self.dimension = dimension
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.embeddings = nn.Embedding(self.num_nodes, self.dimension).to(self.device)
        self.embeddings.weight.data.normal_(0.0, 1./sqrt(dimension))
        self.contexts = nn.Embedding(self.num_nodes, self.dimension).to(self.device)
        self.contexts.weight.data.normal_(0.0, 1./sqrt(dimension))

    def forward(self, u, v, sign):
        emb_u = self.embeddings(u)
        ctx_v = self.contexts(v)
        prod = torch.sum(torch.mul(emb_u, ctx_v), dim=1)
        prod = torch.mul(sign, prod)
        loss = torch.sum(nn.functional.logsigmoid(prod))
        loss = loss.neg()
        return loss

class TripletEmbedding(nn.Module):
    def __init__(self, num_nodes, dimension, device='cuda:0'):
        super().__init__()
        self.num_nodes = num_nodes
        self.dimension = dimension
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.embeddings = nn.Embedding(self.num_nodes, self.dimension).to(self.device)
        self.embeddings.weight.data.normal_(0.0, 1./sqrt(dimension))
        self.contexts = nn.Embedding(self.num_nodes, self.dimension).to(self.device)
        self.contexts.weight.data.normal_(0.0, 1./sqrt(dimension))

    def forward(self, u, v, w):
        emb_u = self.embeddings(u)
        ctx_v = self.contexts(v)
        ctx_w = self.contexts(w)
        pos_prod = torch.sum(torch.mul(emb_u, ctx_v), dim=1)
        neg_prod = torch.sum(torch.mul(emb_u, ctx_w), dim=1)
        loss = torch.sum(nn.functional.logsigmoid(pos_prod-neg_prod))
        loss = loss.neg()
        return loss

class ModelIterator(object):

    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def lr_decay(self):
        self.scheduler.step()

    def get_embeddings(self):
        return self.model.embeddings.weight.data.cpu().numpy()
    
    def set_embeddings(self, emb):
        self.model.embeddings.weight.data.copy_(torch.from_numpy(emb).to(device=self.model.device))

    def get_contexts(self):
        return self.model.contexts.weight.data.cpu().numpy()

    def set_contexts(self, ctx):
        self.model.contexts.weight.data.copy_(torch.from_numpy(ctx).to(device=self.model.device))

class NodeEmbedding(ModelIterator):

    def __init__(self, num_nodes, dimension, learning_rate, device='cuda:0'):
        model = SkipGramNS(num_nodes, dimension, device=torch.device(device if torch.cuda.is_available() else 'cpu'))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        super().__init__(model, optimizer, scheduler)
    
    def feed(self, u, v, sign):
        self.optimizer.zero_grad()
        tu = torch.tensor(u, device=self.model.device, dtype=torch.long)
        tv = torch.tensor(v, device=self.model.device, dtype=torch.long)
        tsign = torch.tensor(sign, device=self.model.device, dtype=torch.float)
        loss = self.model(tu, tv, tsign)
        loss.backward()
        self.optimizer.step()

class TripletNodeEmbedding(ModelIterator):

    def __init__(self, num_nodes, dimension, learning_rate, device='cuda:0'):
        model = TripletEmbedding(num_nodes, dimension, device=torch.device(device if torch.cuda.is_available() else 'cpu'))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        super().__init__(model, optimizer, scheduler)

    def feed(self, u, v, w):
        self.optimizer.zero_grad()
        tu = torch.tensor(u, device=self.model.device, dtype=torch.long)
        tv = torch.tensor(v, device=self.model.device, dtype=torch.long)
        tw = torch.tensor(w, device=self.model.device, dtype=torch.long)
        loss = self.model(tu, tv, tw)
        loss.backward()
        self.optimizer.step()