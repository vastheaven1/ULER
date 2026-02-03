import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer_dffn import TransformerEncoder
from modules.lmsc import LMSC
from src.utils import *
  
np.random.seed(2024) 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") 

# This is for the DEAP dataset.
# For DREAMER and WESAD datasets, only minor adjustments are needed: 
# adjust the values of modalities, channels, 
# and sampling rates at the corresponding positions according to the description in the paper.
# ULER Main Model Class
class ULER_Model(nn.Module):
    def __init__(self, hyp_params):
        super(ULER_Model, self).__init__()
        # Hyperparameters from config
        self.num_heads = hyp_params.num_heads 
        self.layers = hyp_params.layers 
        self.attn_dropout = hyp_params.attn_dropout 
        self.relu_dropout = hyp_params.relu_dropout 
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout 
        self.category = hyp_params.category
        self.attn_mask = False
        
        # Embedding dimension
        self.embed_dim = 30   
        # LMSC (Lightweight Multi-Scale Convolution) for each modality.
        # This is for the DEAP dataset.
        self.lmsc1 = LMSC(128, 30)  
        self.lmsc2 = LMSC(128, 30)  
        self.lmsc3 = LMSC(128, 30) 
        self.lmsc4 = LMSC(128, 30)  

        self.proj_all = nn.Sequential(
                    nn.Conv1d(37, 30, kernel_size=1, padding=0, bias=True),
                    nn.BatchNorm1d(30),
                    nn.Dropout(self.out_dropout))
        
        
        # Learnable weights for LJA weighted fusion
        self.weight_intra = nn.Parameter(torch.ones(1))  
        self.weight_inter = nn.Parameter(torch.ones(1))  

        nn.init.constant_(self.weight_intra, 1.0)
        nn.init.constant_(self.weight_inter, 1.0)
        
        # Intra-LJA Transformers (intra-modal attention)
        self.intra_lja_m1 = self.get_network(layers=2)
        self.intra_lja_m2 = self.get_network(layers=2)
        self.intra_lja_m3 = self.get_network(layers=2)
        self.intra_lja_m4 = self.get_network(layers=2)
        
        # Inter-LJA Transformers (cross-modal attention)
        self.inter_lja_m1 = self.get_network(layers=2)
        self.inter_lja_m2 = self.get_network(layers=2)
        self.inter_lja_m3 = self.get_network(layers=2)
        self.inter_lja_m4 = self.get_network(layers=2)

        # Final classification layers
        self.fc_module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30 * 30, 64),
            nn.BatchNorm1d(64),  
            nn.Dropout(self.out_dropout)
        )

        self.out_put = nn.Linear(64, self.category)
    def get_network(self, layers=-1):
        
        return TransformerEncoder(embed_dim=self.embed_dim, 
                                  num_heads=self.num_heads, 
                                  layers=max(self.layers, layers),
                                  attn_dropout=self.attn_dropout, 
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout, 
                                  embed_dropout=self.embed_dropout, 
                                  attn_mask=self.attn_mask
                                  )       

     
            
    def forward(self,m1_input,m2_input,m3_input,m4_input):    

        # Stage 1: LMSC Feature Extraction
        m1_features = self.lmsc1(m1_input.permute(0, 2, 1)).permute(2, 0, 1)  
        m2_features = self.lmsc2(m2_input.permute(0, 2, 1)).permute(2, 0, 1)  
        m3_features = self.lmsc3(m3_input.permute(0, 2, 1)).permute(2, 0, 1)  
        m4_features = self.lmsc4(m4_input.permute(0, 2, 1)).permute(2, 0, 1)  
        
        # Prepare features for Inter-LJA (cross-modal attention)
        other_for_m1 = torch.cat([m2_features,m3_features,m4_features],dim=0) 
        other_for_m2 = torch.cat([m1_features,m3_features,m4_features],dim=0) 
        other_for_m3 = torch.cat([m1_features,m2_features,m4_features],dim=0) 
        other_for_m4 = torch.cat([m1_features,m2_features,m3_features],dim=0) 

        # Stage 2: LJA-based Multimodal Fusion
        # Intra-LJA: Intra-modal attention
        intra_m1 = self.intra_lja_m1(m1_features)  
        intra_m2 = self.intra_lja_m2(m2_features)   
        intra_m3 = self.intra_lja_m3(m3_features)   
        intra_m4 = self.intra_lja_m4(m4_features)   
        
        # Inter-LJA: Cross-modal attention
        inter_m1 = self.inter_lja_m1(m1_features,other_for_m1,other_for_m1) 
        inter_m2 = self.inter_lja_m2(m2_features,other_for_m2,other_for_m2)  
        inter_m3 = self.inter_lja_m3(m3_features,other_for_m3,other_for_m3) 
        inter_m4 = self.inter_lja_m4(m4_features,other_for_m4,other_for_m4)  
       
        fusion_weights = F.softmax(torch.cat([self.weight_intra, self.weight_inter]), dim=0)
       
        intra_all = torch.cat([intra_m1, intra_m2, intra_m3, intra_m4], dim=0)
        inter_all = torch.cat([inter_m1, inter_m2, inter_m3, inter_m4], dim=0)
        fused_features = fusion_weights[0] * intra_all + fusion_weights[1] * inter_all
      
        conv_output = self.proj_all(fused_features.permute(1,0,2)) 
        fc_features = self.fc_module(conv_output)  
        output = self.out_put(fc_features) 
   
        return output, fc_features
