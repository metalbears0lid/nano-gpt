import torch
from torch import nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many sequences to process in parallel
block_size = 8  # aka context length
n_embd = 32     # dimension of embedded tokens
max_iters = 5000
eval_interval = 500
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


torch.manual_seed(1337)

#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# map from chars to ints and vice versa
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Takes embedded sequences of tokens (B, T, C) and combines information from previous tokens via self-attention head.
# (B, T, C) -> (B, T, H)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)   # what I'm looking for
        self.key = nn.Linear(n_embd, head_size, bias=False)     # what I contain
        self.value = nn.Linear(n_embd, head_size, bias=False)   # what I'll communicate if another token finds me interesting
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        # x is (B, T, C)
        B, T, C = x.shape
        # construct the (T, T) matrix of weights
        q = self.query(x)   # (B, T, H)
        k = self.key(x)     # (B, T, H)
        v = self.value(x)   # (B, T, H)

        wei = q @ torch.transpose(k, -2, -1) * self.head_size**(-0.5)   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, -torch.inf)         # need to restrict to :T because at generation, we'll need to predict from shorter token seqs
        wei = torch.softmax(wei, dim=-1)

        out = wei @ v   # (B, T, H)
        return out


# Propagate num_heads heads, each having head_size dimensions.  Concatenates results to form output of multi-head
# (B, T, C) => (B, T, num_heads * head_size)
class MultiHeadAttention(nn.Module):
    '''Multiple heads of self-attention in parallel'''
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)   # project back into the residual pathway
    
    def forward(self, x):
        # x is (B, T, C)
        # h(x) is (B, T, head_size)
        # output is (B, T, num_heads * head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
        
# Feed forward layer that allows tokens to "think" after self-attending
class FeedForward(nn.Module):
    '''Linear layer followed by a non-linearity'''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # project back into the residual pathway
        )
    
    def forward(self, x):
        return self.net(x)

# Block of multi-head attention with feed forward, which will be repeated sequentially in our LLM
class Block(nn.Module):
    '''Transformer block: communication followed by computation'''
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.sa_heads(x)    # self-attention with residual/skip connection
        x = x + self.ffwd(x)        # feed-forward with residual/skip connection
        return x

# simple bigram model (next char only depends on prev char)
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads=4) for _ in range(3)])
        self.lm_head = nn.Linear(n_embd, vocab_size)    # language model head takes embeddings to logits
    
    def forward(self, idx, targets=None):
        # idx and targets are (B, T) tensors of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)   # (B, T, C) batch(4) x time(8) x channel(n_embd=32)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb       # (B, T, C)
        x = self.blocks(x)          # (B, T, n_embd=32)
        logits = self.lm_head(x)    # (B, T, vocab_size=65)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T).  
        # For each batch, generate next predicted token to get (B, T+1).  
        # Do this max_new_tokens times.
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens
            # idx is the complete sequence of generated tokens.  idx_cond is the most recent sequence of tokens of len at most block_size
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)                            # get preds.  logits is (B, T, C)
            logits = logits[:, -1, :]                           # focus only on last time step (B, C)
            probs = F.softmax(logits, dim=-1)                   # apply softmax to get probabilities (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
            
        return idx

    
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)   # typically 3e-4 is a good learning rate for more complex problems

# train the model
for iter in range(max_iters):
    # every once in a while, evaluate the loss on train/val sets
    if iter % eval_interval == 0 or iter==max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# general from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(idx=context, max_new_tokens=100)[0].tolist())

print(generated_chars)

