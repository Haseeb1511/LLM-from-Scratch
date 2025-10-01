# Decoder only transformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from math import log
from torch.nn import functional as F



torch.manual_seed(1337)
batch_size = 64
block_size = 256 # Maximum context length for prediction?  no of token
max_iter = 5000
eval_iter = 200
learning_rate = 3e-4
eval_interval = 500
n_embed = 384 # Embedding Dimensions
n_head = 6
n_layer = 6
dropout = 0.25

device  = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt

with open("input.txt","r") as f:
  text = f.read()


char  = sorted(list(set(text)))
vocab_size = len(char)


# Creating a mapping for character
# we are creating dictionary to which we gave character and it give index
stoi = {ch:i for i,ch in enumerate(char)} #create dictionary with charater and its index--->{"a":1}
# we are creating dictionary to which we gave index and it give the corresponding character
#x is a PyTorch tensor, but itos expects a Python integer index, not a tensor object.
itos = {i:ch for i,ch in enumerate(char)}  # create a dictionary with index and character -->{1:"a"}

encode = lambda x : [stoi[c] for c in x]
decode = lambda x :"".join([itos[i] for i in x])


# split the data for training and testing
data = torch.tensor(encode(text),dtype=torch.long)  # Encode the data
n = int(0.9*len(data))
train_data = data[:n]  # 90% data for training--># elements from start (index 0) up to n-1
val_data = data[n:]  # last 10% for validation data---># elements from index n to the end


def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data)-block_size,(batch_size,)) # Pick random starting indices
  # We used stack to get esult shape = (batch_size, block_size).
  # Withput stack we get single dimension 1D tensor with stack
  # Build inputs x
  x = torch.stack([data[i:i+block_size] for i in ix]) # input for model like ==> 47,10,30
  # Build output y
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #output for model like ===> 10,30,
  return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train","val"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            x,y = get_batch(split) #get batch using get_batch function
            logit,loss = model(x,y) #pass the batch to model function
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.query_vector = nn.Linear(n_embed,head_size,bias=False)
        self.key_vector = nn.Linear(n_embed,head_size,bias=False)
        self.value_vector = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        q = self.query_vector(x)
        k = self.key_vector(x)
        v = self.value_vector(x)
        # Compute attention score
        wei = q @ k.transpose(-2,-1)*C**-0.5  # scaling & c=dimesnion of vector  (B,T,C) @ (B,C,T).T==(B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0 ,float("-inf")) #(B,T,T)
        wei = F.softmax(wei,dim=-1) # (B,T,T)
        # Dot product of attention score with the value vector
        out = wei @ v  #(B,T,T) @ (B,T,C)===> (B,T,C)
        return out
    

    
class MultiHeadAttetnion(nn.Module):
    """
Args:
    num_heads (int): Number of parallel attention heads. Each head is its own self-attention block.
        Multiple heads allow the model to capture different relationships or “representation
        subspaces” of the input sequence at the same time.

    head_size (int): Dimensionality of each attention head’s output vector. Inside each head,
        the input embedding is projected into query, key, and value vectors of size head_size.

Example:
    n_embed = 32
    num_heads = 4
    head_size = 8

    - Each token embedding has 32 dimensions.
    - Each head applies Linear(32 → 8) to produce q, k, v vectors of size 8.
    - One head outputs (B, T, 8). With 4 heads, we get 4 such outputs.
    - Concatenating across heads gives (B, T, 32).

Notes:
    - Each head captures a different aspect of the token’s meaning.
    - Concatenation restores the original embedding size (32), ensuring consistent
      dimensionality across layers.
    - If num_heads * head_size ≠ n_embed, the multi-head output size will not match
      the input size. In standard transformers, a final Linear projection is used
      to resolve this mismatch.

    embedding = 32
    num_head = 4
    head_output = 8
    mean 4 head will have input of 32 dimension and converted to 8 dimesnion q,k,v vector and at the end we will have 4 output of 8 dimension we will concatenate it and get 2 dimesnion final output of multi head attention
    if embedding size and final output dimension is not sam transformer will fail

  """
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Each head learned independently, so you need a final mixing layer to let them interact. so we add a linear layer to let them interact
        # It takes the concatenated output (B, T, n_embed) and projects it back into the embedding space.
        # This way, the output of multi-head attention has the same dimension as the input embedding.
        # This is necessary so you can:
        # Add residual connections (x + attention_out)
        # Stack more transformer blocks without shape mismatch.
        self.proj = nn.Linear(n_embed,n_embed)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1) # concatinating over the C Dimension    (B,T,C)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),# First layer expands the embedding dimension (hidden_dim, usually n_embed).
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed), # Second layer projects back to n_embed so the shape matches the input (needed for residual connections).
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)



class Block(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size = n_embed//n_head   # 32/4==>8
        self.sa = MultiHeadAttetnion(num_heads=n_head,head_size=head_size)
        self.ffd = FeedForward(n_embed=n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        # Self-attention with residual connection
        x = x + self.sa(self.ln1(x))  
        # Feed-forward with residual connection
        x = x + self.ffd(self.ln2(x))  
        return x



# B = batch size
# T = sequence length (context length we fed in)
# V = vocab size (number of possible tokens)
class Bigramlanguagemodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table = nn.Embedding(block_size,n_embed)  

        self.block = nn.Sequential(*[Block(n_embed,n_head=n_head) for _ in range(n_layer)]) # how many layer we want
    
        self.ln_F = nn.LayerNorm(n_embed)  #fianl layer norm
        self.lm_head = nn.Linear(n_embed,vocab_size)


    def forward(self,idx,target=None): # idx is the input text, already tokenized and converted to integers from your vocabulary.
        B,T = idx.shape
        token_emb = self.token_embedding_table(idx)  # ==> (B,T,C)==>(4,8,vocab_size)==>torch.Size([4, 8, 65])
        pos_embedding = self.position_embedding_table(torch.arange(T,device=device))  # (T,C)
        x = token_emb + pos_embedding # Sum the postion embedding + token embedding  ==> (B,T,C)
        x = self.block(x)
        logits = x=self.lm_head(x) # (B,T,vocab_szie)
        loss=None
        if target is not None:
            #RuntimeError: Expected target size [4, 65], got [4, 8]
            # loss function accept (B,C)
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)
        return logits,loss


    def generate(self,idx,max_new_token):
        # it take (B,T) and make it ==> (B,T+1,T+2...)
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_token):#get the predeiction
            #crop idx to last block size token
            idx_cond = idx[:,-block_size:]    
            logits,loss = self(idx_cond)  # self is forward function
            #focus on the last time step
            # : → keep all batches (B)
            # -1 → only take the last time step (T-1)
            # : → keep all vocab logits (C)
            logits = logits[:,-1,:] # become (B,C)
            #apply softmax to get probability
            probs = F.softmax(logits,dim=-1) #(B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sample index to running samples
            idx = torch.cat((idx,idx_next),dim=1) # (B,T+1)
        return idx
  

       

model = Bigramlanguagemodel() # Model object
optimizer = torch.optim.AdamW(model.parameters(),learning_rate)

for iter in range(max_iter):
    if iter % eval_interval==0:   # iter % eval_interval → the remainder when iter is divided by eval_interval.  0 mean "no remainder"
        losses = estimate_loss()
        print(f"step {iter}:train loss {losses["train"]:.4f} val loss {losses["val"]:.4f}")

        xb,yb = get_batch("train")

        optimizer.zero_grad()  #clear gradient accumulation
        logits,loss = model(xb,yb)  #pass the input to model
        loss.backward()#backpropogation
        optimizer.step() #update parameters


    
# Generate from the model
context = torch.zeros((1,1),dtype=torch.long,device=device)
output = model.generate(idx=context, max_new_token=2000)
decoded = decode(output[0].tolist())  # convert tensor → list[int]
print(decoded)
