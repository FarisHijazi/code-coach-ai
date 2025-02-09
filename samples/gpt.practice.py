"""
Code from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = #PRACTICE-1: determine the device for tensor computations --> HINT: -->                                          HINT: consider how to handle device allocation for tensors --> SOLUTION: -->                                          SOLUTION: Chunk 1: device = 'cuda' if torch.cuda.is_available() else 'cpu'eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
#PRACTICE-2: calculate the size of the vocabulary from the unique characters --> HINT: -->                                          HINT: think about how the vocabulary size relates to the characters in the text --> SOLUTION: -->                                          SOLUTION: Chunk 2: vocab_size = len(chars)# create a mapping from characters to integers
#PRACTICE-2: create a mapping from characters to integers --> HINT: -->                                          HINT: consider how you would map characters to unique integer values --> SOLUTION: -->                                          SOLUTION: Chunk 2: stoi = { ch:i for i,ch in enumerate(chars) }itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = #PRACTICE-3: generate random indices for batch sampling --> HINT: -->                                          HINT: think about how to sample indices for creating batches --> SOLUTION: -->                                          SOLUTION: Chunk 3: ix = torch.randint(len(data) - block_size, (batch_size,))    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    #PRACTICE-4: Set the model to evaluation mode --> HINT: -->                                          HINT: Understand the difference between training and evaluation modes --> SOLUTION: -->                                          SOLUTION: Chunk 4: model.eval()    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = #PRACTICE-4: Store the loss for each iteration --> HINT: -->                                          HINT: You need to understand how to accumulate losses for evaluation --> SOLUTION: -->                                          SOLUTION: Chunk 4: losses[k] = loss.item()        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #PRACTICE-5: This line initializes a lower triangular matrix for attention masking. --> HINT: -->                                          HINT: Consider how attention mechanisms use masks to prevent attending to future tokens. --> SOLUTION: -->                                          SOLUTION: Chunk 5: self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = #PRACTICE-6: apply dropout to the attention weights --> HINT: -->                                          HINT: consider the purpose of dropout in this context --> SOLUTION: -->                                          SOLUTION: Chunk 6: wei = self.dropout(wei)        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        #PRACTICE-7: Implement the projection layer in the forward method. --> HINT: -->                                          HINT: Think about how to combine the outputs of the attention heads. --> SOLUTION: -->                                          SOLUTION: Chunk 7: self.proj = nn.Linear(head_size * num_heads, n_embd)        #PRACTICE-7: Implement the dropout functionality in the forward method. --> HINT: -->                                          HINT: Consider how dropout is typically applied in neural networks. --> SOLUTION: -->                                          SOLUTION: Chunk 7: self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = #PRACTICE-8: Concatenate the outputs from all attention heads. --> HINT: -->                                          HINT: Think about how multi-head attention combines information from different heads. --> SOLUTION: -->                                          SOLUTION: Chunk 8: out = torch.cat([h(x) for h in self.heads], dim=-1)        out = #PRACTICE-8: Apply dropout after the projection layer. --> HINT: -->                                          HINT: Understand the role of dropout in preventing overfitting. --> SOLUTION: -->                                          SOLUTION: Chunk 8: out = self.dropout(self.proj(out))        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        #PRACTICE-9: Implement the forward pass for the FeedForward layer. --> HINT: -->                                          HINT: Think about how to apply the layers defined in the __init__ method. --> SOLUTION: -->                                          SOLUTION: Chunk 9: return self.net(x)
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        #PRACTICE-9: Create an instance of the FeedForward class. --> HINT: -->                                          HINT: Understand how the FeedForward layer is integrated into the Block. --> SOLUTION: -->                                          SOLUTION: Chunk 9: self.ffwd = FeedFoward(n_embd)        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + #PRACTICE-10: Implement the residual connection manually. --> HINT: -->                                          HINT: Think about how to add the output of self.sa to x. --> SOLUTION: -->                                          SOLUTION: Chunk 10: x = x + self.sa(self.ln1(x))        x = x + #PRACTICE-10: Implement the residual connection manually. --> HINT: -->                                          HINT: Consider how to add the output of self.ffwd to x. --> SOLUTION: -->                                          SOLUTION: Chunk 10: x = x + self.ffwd(self.ln2(x))        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        #PRACTICE-11: Initialize weights of the model. --> HINT: -->                                          HINT: Consider how weight initialization affects model training. --> SOLUTION: -->                                          SOLUTION: Chunk 11: self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #PRACTICE-11: Initialize the weights of linear layers. --> HINT: -->                                          HINT: Think about the implications of weight initialization on convergence. --> SOLUTION: -->                                          SOLUTION: Chunk 11: torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            #PRACTICE-11: Initialize the weights of linear layers. --> HINT: -->                                          HINT: Think about the implications of weight initialization on convergence. --> SOLUTION: -->                                          SOLUTION: Chunk 11: torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = #PRACTICE-13: calculate the loss using cross-entropy --> HINT: -->                                          HINT: Understand how to reshape logits and targets for loss calculation --> SOLUTION: -->                                          SOLUTION: Chunk 13: loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
#PRACTICE-15: calculate and display the number of parameters in the model --> HINT: -->                                          HINT: Consider how to compute the total number of parameters in a PyTorch model. --> SOLUTION: -->                                          SOLUTION: Chunk 15: print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# create a PyTorch optimizer
#PRACTICE-15: create an optimizer for the model parameters --> HINT: -->                                          HINT: Think about how to set up an optimizer in PyTorch for training a model. --> SOLUTION: -->                                          SOLUTION: Chunk 15: optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.#PRACTICE-16: Implement the gradient zeroing step without using the set_to_none argument. --> HINT: -->                                          HINT: Think about how to reset gradients in PyTorch. --> SOLUTION: -->                                          SOLUTION: Chunk 16: optimizer.zero_grad(set_to_none=True)    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#PRACTICE-17: Replace this line with your own code to visualize the generated text. --> HINT: -->                                          HINT: Focus on understanding how to use the generate method and decode the output. --> SOLUTION: -->                                          SOLUTION: Chunk 17: print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
