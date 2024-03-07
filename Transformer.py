import torch
import math

# class for self-attention calculation from a single head
# take input x, project to Q,K,V, apply the self-attention formula (Q @ K.T / sqrt(head_size)) @ V
# in addition, the class takes an input decoder - it signals whether it is a encoder head or decoder head, decoder head has an additional mask step
# input: [B, T, C] --> [B, T, H], where for multi-head attention, H = C / num_heads, C = emb_dim
class Head(torch.nn.Module):
    def __init__(self,emb_dim,head_size,block_size,dropout_rate,is_decoder):
        super().__init__()
        self.H = head_size
        self.key = torch.nn.Linear(emb_dim,head_size,bias=False) # not including bias because of layer norm include bias term
        self.query = torch.nn.Linear(emb_dim,head_size,bias=False)
        self.value = torch.nn.Linear(emb_dim,head_size,bias=False)
        if is_decoder:
            self.register_buffer("tril_mat", torch.tril(torch.ones(block_size,block_size))) # parameters not being updated
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.is_decoder = is_decoder
        
    def forward(self,x):
        B, T, C = x.shape # decompose dimensions: Batch, Time, Embedding
        k = self.key(x) # B, T, H
        q = self.query(x) # B, T, H
        attention_W = q @ k.transpose(-2, -1) * self.H**-0.5 # B, T, T
        if self.is_decoder:
            attention_W = attention_W.masked_fill(self.tril_mat==0, float('-inf')) # B, T, T
        attention_W = torch.nn.functional.softmax(attention_W,dim=-1)  # B, T, T
        attention_W = self.dropout(attention_W)
        v = self.value(x) # B, T, H
        output = attention_W @ v # B, T, H
        return output
    
# class for orchestrate multiple heads for multiple self-attention calculation
# input: [B, T, C] --> output: [B, T, C]
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_dim, num_heads, head_size, block_size,dropout_rate, is_decoder):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(emb_dim,head_size,block_size,dropout_rate,is_decoder) for _ in range(num_heads)])
        self.projection = torch.nn.Linear(emb_dim,emb_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
    def forward(self,x):
        output = torch.cat([h(x) for h in self.heads],dim=-1)
        output = self.projection(output)
        output = self.dropout(output)
        return output
    

# feedforward network after multi-head attention: multiplier parameter is the number of times of neurons in hidden layer than input, default is 4 from the paper
# input [B, T, C] --> output [B, T, C]
class FeedForward(torch.nn.Module):
    def __init__(self,emb_dim,dropout_rate,multiplier):
        super().__init__()
        self.fflayer = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, multiplier * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(multiplier * emb_dim, emb_dim),
            torch.nn.Dropout(p=dropout_rate)
        )
    def forward(self, x):
        output = self.fflayer(x)
        return output


# class for a transformer block: multi-head attention + feed forward + layer norm + residual connection
class TransformerBlock(torch.nn.Module):
    def __init__(self,emb_dim,num_heads,block_size,dropout_rate_attention,dropout_rate_ff,is_decoder,ff_multiplier):
        super().__init__()
        assert emb_dim % num_heads == 0, "number of heads must be divisible by embedding dimention to determine the head size"
        head_size = emb_dim // num_heads
        self.multi_atten = MultiHeadAttention(emb_dim = emb_dim, num_heads = num_heads, head_size = head_size, block_size = block_size,
                                              dropout_rate = dropout_rate_attention, is_decoder = is_decoder)
        self.feedforward = FeedForward(emb_dim = emb_dim, dropout_rate = dropout_rate_ff,multiplier = ff_multiplier)
        self.layernorm1 = torch.nn.LayerNorm(emb_dim)
        self.layernorm2 = torch.nn.LayerNorm(emb_dim)
    
    def forward(self, x):
        output = x + self.multi_atten(self.layernorm1(x)) # layer norm + multi-head attention + residual
        output =  output + self.feedforward(self.layernorm2(output)) # layer norm + feed forward + residual
        return output

# positional encoding from the paper
class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_dim, block_size,dropout_rate):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        # create common divisers: 1 / n^(2i/emb_dim) where i = 0, ..., (emb_dim/2)-1
        #  with some manipulation, this can be exp(-2i*log(n)/emb_dim)
        div_term = torch.exp(-torch.arange(0, emb_dim, 2)*math.log(10000)/emb_dim) # shape (emb_dim/2,)
        # k = 0, ..., block_size-1, column vector
        k = torch.arange(block_size).unsqueeze(1) # shape [block_size,1]
        # create calculation between sin and cos
        product_term = k * div_term # shape [block_size,emb_dim/2]

        # initialize PE matrix
        pe = torch.zeros((block_size,emb_dim))
        # assign values 
        pe[:,0::2] = torch.sin(product_term)
        pe[:,1::2] = torch.cos(product_term)
        # create batch dimension
        pe = pe.unsqueeze(0)

        # make the parameters untrainable
        self.register_buffer("pos_enc", pe)

    def forward(self, x):
        output = x + self.pos_enc
        return self.dropout(output)


# create transformer!!!
class TransformerClass(torch.nn.Module):
    def __init__(self,vocab_size,emb_dim,n_layer,num_heads,block_size,dropout_rate_attention,dropout_rate_ff,dropout_rate_pos_enc,is_decoder=True,ff_multiplier=4):
        super().__init__()
        # embedding layer
        self.embedding = torch.nn.Embedding(vocab_size,emb_dim)
        # positional encoding
        self.positional_encoding = PositionalEncoding(emb_dim = emb_dim, block_size = block_size, dropout_rate = dropout_rate_pos_enc)
        # transformer block
        self.blocks = torch.nn.Sequential(*[TransformerBlock(emb_dim,num_heads,block_size,dropout_rate_attention,dropout_rate_ff,is_decoder,ff_multiplier)
                                            for _ in range(n_layer)])
        self.final_layernorm = torch.nn.LayerNorm(emb_dim)
        self.final_linear = torch.nn.Linear(emb_dim,vocab_size)
    
    def forward(self,x):
        # x shape: [B, T]
        x_emb = self.embedding(x) # shape: [B, T, C]
        x_pos = self.positional_encoding(x_emb) # shape: [B, T, C]
        x = self.blocks(x_pos) # shape: [B, T, C]
        x = self.final_layernorm(x) # shape: [B, T, C]
        logit = self.final_linear(x) # shape: [B, T, vocab_size]
        return logit

