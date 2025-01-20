from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

# Enable cuDNN benchmarking and deterministic algorithms
cudnn.benchmark = True
cudnn.deterministic = False

class SkipGramNS(nn.Module):
    def __init__(self, num_nodes, dimension, device='cuda:0'):
        super().__init__()
        self.num_nodes = num_nodes
        self.dimension = dimension
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Use float16 for embeddings to reduce memory usage and increase speed
        self.embeddings = nn.Embedding(self.num_nodes, self.dimension, dtype=torch.float16).to(self.device)
        self.embeddings.weight.data.normal_(0.0, 1./sqrt(dimension))
        self.contexts = nn.Embedding(self.num_nodes, self.dimension, dtype=torch.float16).to(self.device)
        self.contexts.weight.data.normal_(0.0, 1./sqrt(dimension))

    def forward(self, u, v, sign):
        emb_u = self.embeddings(u)
        ctx_v = self.contexts(v)
        prod = torch.sum(torch.mul(emb_u, ctx_v), dim=1)
        prod = torch.mul(sign, prod)
        loss = torch.sum(nn.functional.logsigmoid(prod))
        return loss.neg()

class TripletEmbedding(nn.Module):
    def __init__(self, num_nodes, dimension, device='cuda:0'):
        super().__init__()
        self.num_nodes = num_nodes
        self.dimension = dimension
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Use float16 for embeddings
        self.embeddings = nn.Embedding(self.num_nodes, self.dimension, dtype=torch.float16).to(self.device)
        self.embeddings.weight.data.normal_(0.0, 1./sqrt(dimension))
        self.contexts = nn.Embedding(self.num_nodes, self.dimension, dtype=torch.float16).to(self.device)
        self.contexts.weight.data.normal_(0.0, 1./sqrt(dimension))

    def forward(self, u, v, w):
        emb_u = self.embeddings(u)
        ctx_v = self.contexts(v)
        ctx_w = self.contexts(w)
        pos_prod = torch.sum(torch.mul(emb_u, ctx_v), dim=1)
        neg_prod = torch.sum(torch.mul(emb_u, ctx_w), dim=1)
        loss = torch.sum(nn.functional.logsigmoid(pos_prod-neg_prod))
        return loss.neg()

class ModelIterator(object):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler()  # Add gradient scaler for mixed precision

    def lr_decay(self):
        self.scheduler.step()

    def get_embeddings(self):
        with torch.no_grad():
            return self.model.embeddings.weight.data.cpu().float().numpy()
    
    def set_embeddings(self, emb):
        with torch.no_grad():
            self.model.embeddings.weight.data.copy_(torch.from_numpy(emb).to(device=self.model.device, dtype=torch.float16))

    def get_contexts(self):
        with torch.no_grad():
            return self.model.contexts.weight.data.cpu().float().numpy()

    def set_contexts(self, ctx):
        with torch.no_grad():
            self.model.contexts.weight.data.copy_(torch.from_numpy(ctx).to(device=self.model.device, dtype=torch.float16))

class NodeEmbedding(ModelIterator):
    def __init__(self, num_nodes, dimension, learning_rate, device='cuda:0'):
        model = SkipGramNS(num_nodes, dimension, device=device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        super().__init__(model, optimizer, scheduler)
    
    def feed(self, u, v, sign, batch_size=1024):
        self.optimizer.zero_grad()
        
        # Process data in batches
        for i in range(0, len(u), batch_size):
            batch_u = torch.tensor(u[i:i+batch_size], device=self.model.device, dtype=torch.long)
            batch_v = torch.tensor(v[i:i+batch_size], device=self.model.device, dtype=torch.long)
            batch_sign = torch.tensor(sign[i:i+batch_size], device=self.model.device, dtype=torch.float16)
            
            # Use gradient scaling for mixed precision training
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = self.model(batch_u, batch_v, batch_sign)
            
            self.scaler.scale(loss).backward()
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

class TripletNodeEmbedding(ModelIterator):
    def __init__(self, num_nodes, dimension, learning_rate, device='cuda:0'):
        model = TripletEmbedding(num_nodes, dimension, device=device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        super().__init__(model, optimizer, scheduler)

    def feed(self, u, v, w, batch_size=1024):
        self.optimizer.zero_grad()
        
        # Process data in batches
        for i in range(0, len(u), batch_size):
            batch_u = torch.tensor(u[i:i+batch_size], device=self.model.device, dtype=torch.long)
            batch_v = torch.tensor(v[i:i+batch_size], device=self.model.device, dtype=torch.long)
            batch_w = torch.tensor(w[i:i+batch_size], device=self.model.device, dtype=torch.long)
            
            # Use gradient scaling for mixed precision training
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = self.model(batch_u, batch_v, batch_w)
            
            self.scaler.scale(loss).backward()
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

# Memory management utilities
def clear_gpu_memory():
    torch.cuda.empty_cache()

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()