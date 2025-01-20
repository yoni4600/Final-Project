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
        self.embeddings = nn.Embedding(self.num_nodes, self.dimension, device=self.device)
        self.embeddings.weight.data.normal_(0.0, 1./sqrt(dimension))
        self.contexts = nn.Embedding(self.num_nodes, self.dimension, device=self.device)
        self.contexts.weight.data.normal_(0.0, 1./sqrt(dimension))

    @torch.cuda.amp.autocast()
    def forward(self, u, v, sign):
        emb_u = self.embeddings(u)
        ctx_v = self.contexts(v)
        prod = torch.sum(torch.mul(emb_u, ctx_v), dim=1)
        prod = torch.mul(sign, prod)
        loss = torch.sum(nn.functional.logsigmoid(prod))
        return loss.neg()

class ModelIterator(object):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler()  # For mixed precision training

    def lr_decay(self):
        self.scheduler.step()

    @torch.no_grad()
    def get_embeddings(self):
        return self.model.embeddings.weight.data.cpu().numpy()
    
    def set_embeddings(self, emb):
        with torch.no_grad():
            self.model.embeddings.weight.data.copy_(torch.from_numpy(emb).to(device=self.model.device))

    @torch.no_grad()
    def get_contexts(self):
        return self.model.contexts.weight.data.cpu().numpy()

    def set_contexts(self, ctx):
        with torch.no_grad():
            self.model.contexts.weight.data.copy_(torch.from_numpy(ctx).to(device=self.model.device))

class NodeEmbedding(ModelIterator):
    def __init__(self, num_nodes, dimension, learning_rate, device='cuda:0'):
        model = SkipGramNS(num_nodes, dimension, device=device)
        # Use Adam optimizer with better default settings for embeddings
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        super().__init__(model, optimizer, scheduler)
    
    def feed(self, u, v, sign, batch_size=10000):
        # Process data in batches to avoid OOM
        for i in range(0, len(u), batch_size):
            batch_u = torch.tensor(u[i:i+batch_size], device=self.model.device, dtype=torch.long)
            batch_v = torch.tensor(v[i:i+batch_size], device=self.model.device, dtype=torch.long)
            batch_sign = torch.tensor(sign[i:i+batch_size], device=self.model.device, dtype=torch.float)
            
            self.optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast():
                loss = self.model(batch_u, batch_v, batch_sign)
            
            # Scale loss and backpropagate
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

# Example usage with optimized batch processing
def train_embeddings(node_embedding, data, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        # Assume data is a tuple of (u, v, sign)
        node_embedding.feed(data[0], data[1], data[2])
        # Update learning rate based on loss
        node_embedding.lr_decay()