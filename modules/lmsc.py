from torch import nn
import torch.nn.functional as F

# Lightweight Multi-Scale Convolution module (LMSC)
class LMSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LMSC, self).__init__()
        
        # Five parallel convolution branches with different kernel sizes
        self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)
        self.ru0 = nn.GELU()
        self.bn0 = nn.BatchNorm1d(out_channels)
        self.dropout0 = nn.Dropout(0.1) 

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.ru1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, bias=True)
        self.ru2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, bias=True)
        self.ru3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.dropout3 = nn.Dropout(0.1)

        self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4, bias=True)
        self.ru4 = nn.GELU()
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.dropout4 = nn.Dropout(0.1)
        
        self.dropout_all = nn.Dropout(0.1)

    def forward(self, x):
        # Process through each branch
        out0 = self.dropout0(self.ru0(self.bn0(self.conv0(x))))
        out1 = self.dropout1(self.ru1(self.bn1(self.conv1(x))))
        out2 = self.dropout2(self.ru2(self.bn2(self.conv2(x))))
        out3 = self.dropout3(self.ru3(self.bn3(self.conv3(x))))
        out4 = self.dropout4(self.ru4(self.bn4(self.conv4(x))))

        # Sum all branch outputs (multi-scale feature fusion)
        out = out0 + out1 + out2 + out3 + out4
    
        return self.dropout_all(out)