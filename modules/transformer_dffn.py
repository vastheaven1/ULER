import torch
from torch import nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.multihead_attention import MultiheadAttention
import math
# Transformer Encoder with DFFN (Dynamic Feature Fusion Network)
class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super(TransformerEncoder,self).__init__()
        self.dropout = embed_dropout      
        self.attn_dropout = attn_dropout 
        self.embed_dim = embed_dim 
        self.embed_scale = math.sqrt(embed_dim) 
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask 
        self.normalize = True
        self.layers = nn.ModuleList([])
        for layer in range(layers): 
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout, 
                                                res_dropout=res_dropout, 
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k = None, x_in_v = None): 
        
        # Scale embedding
        x =  x_in     
        # Add positional embedding
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   
        x = F.dropout(x, p=self.dropout, training=self.training) 
        
        # Process key and value inputs for cross-attention
        if x_in_k is not None and x_in_v is not None:
            x_k =  x_in_k
            x_v =  x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        # Apply encoder layers
        intermediates = [x] 
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x,x_k,x_v)
            else:
                x = layer(x)
            intermediates.append(x)
        # Apply layer normalization
        if self.normalize:
            x = self.layer_norm(x) 
        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

# Transformer Encoder Layer with DFFN
class TransformerEncoderLayer(nn.Module): 
    
    def __init__(self, embed_dim, num_heads=3, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super(TransformerEncoderLayer,self).__init__()
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
    
        # Multi-head attention
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask 
        self.relu_dropout = relu_dropout 
        self.res_dropout = res_dropout 
        self.normalize_before = True
        
        # Learnable weights for DFFN stage 1 (residual + linear branch + conv branch)
        self.weight_residual1 = nn.Parameter(torch.ones(1))
        self.weight_x1 = nn.Parameter(torch.ones(1))
        self.weight_x2 = nn.Parameter(torch.ones(1))
        nn.init.constant_(self.weight_residual1, 1.0)
        nn.init.constant_(self.weight_x1, 1.0)
        nn.init.constant_(self.weight_x2, 1.0)
    
        # Learnable weights for DFFN stage 2 (residual + linear branch + conv branch)
        self.weight_residual2 = nn.Parameter(torch.ones(1))
        self.weight_x3 = nn.Parameter(torch.ones(1))
        self.weight_x4 = nn.Parameter(torch.ones(1))
        nn.init.constant_(self.weight_residual2, 1.0)
        nn.init.constant_(self.weight_x3, 1.0)
        nn.init.constant_(self.weight_x4, 1.0)
        
        # Bottleneck dimension for parameter efficiency
        bottleneck_dim = max(1, embed_dim // 2)
        
        # DFFN stage 1: Two parallel branches
        self.branch1 = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim), 
            nn.GELU(),
            nn.Dropout(p=relu_dropout),
            nn.Linear(bottleneck_dim, embed_dim),
            nn.Dropout(p=relu_dropout)
        )
        self.branch2 = nn.Sequential( 
            nn.Conv1d(in_channels=embed_dim, out_channels=bottleneck_dim, kernel_size=1, padding=0, bias=True),
            nn.GELU(),
            nn.Dropout(p=relu_dropout),
            nn.Conv1d(in_channels=bottleneck_dim, out_channels=embed_dim, kernel_size=1, padding=0, bias=True),
            nn.Dropout(p=relu_dropout)
        )
        # DFFN stage 2: Two parallel branches
        self.branch3 = nn.Sequential(   
            nn.Linear(self.embed_dim,  bottleneck_dim ),
            nn.GELU(),
            nn.Dropout(p=relu_dropout),
            nn.Linear( bottleneck_dim,  self.embed_dim),
            nn.Dropout(p=relu_dropout)
          
        )
        self.branch4 = nn.Sequential( 
            nn.Conv1d(in_channels=embed_dim, out_channels=bottleneck_dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Dropout(p=relu_dropout),
            nn.Conv1d(in_channels=bottleneck_dim, out_channels=embed_dim, kernel_size=3, padding=1, bias=True),
            nn.Dropout(p=relu_dropout)
        )
        # Layer normalization for each stage
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(3)]) 
      
    def forward(self, x, x_k=None, x_v=None):
        # Self-attention or cross-attention
        mask =None
        residual = x  
       
        if x_k is None and x_v is None: 
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask) 
        else:
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask) 
        x = F.dropout(x, p=self.res_dropout, training=self.training) 
        x = residual + x # Residual connection
        x = self.layer_norms[0](x)  # Layer norm
        
        # DFFN Stage 1: Parallel linear and convolutional branches
        residual = x  
        x1 = self.branch1(x)
        x2 = self.branch2(x.permute(1,2,0)).permute(2,0,1)  
        weights = F.softmax(torch.cat([self.weight_residual1, self.weight_x1, self.weight_x2]),dim=0)
        x = weights[0] * residual + weights[1] * x1 + weights[2] * x2
        x = self.layer_norms[1](x) 
        
        # DFFN Stage 2: Parallel linear and convolutional branches
        residual = x
        x3 = self.branch3(x)   
        x4 = self.branch4(x.permute(1,2,0)).permute(2,0,1) 
        weights2 = F.softmax(torch.cat([self.weight_residual2, self.weight_x3, self.weight_x4]),dim=0)
        x = weights2[0] * residual + weights2[1] * x3 + weights2[2] * x4
        x = self.layer_norms[2](x) 
        return x     