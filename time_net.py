import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, hyperparas):
        '''
        hyperparas = {'hidden_dim':64,'block_layer_nums':3,'dropout_rate':0.5}
        '''
        super(ResNetBlock, self).__init__()
        
        self.hidden_dim = hyperparas['hidden_dim']
        self.block_layer_nums =hyperparas['block_layer_nums']
        # Define layers for the function f (MLP)
        self.layers = nn.ModuleList()
        self.dropout_rate = hyperparas['dropout_rate']
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        for _ in range(self.block_layer_nums - 1):  # -2 because we already added one layer and last layer is already defined
            self.layers.append(nn.Linear(self.hidden_dim,self.hidden_dim ))
        
        # Layer normalization
        self.layernorms = nn.ModuleList()
        for _ in range(self.block_layer_nums - 1):  # -1 because layer normalization is not applied to the last layer
            self.layernorms.append(nn.LayerNorm(self.hidden_dim))
        
    def forward(self, x):
        # Forward pass through the function f (MLP)
        out = x
        for i in range(self.block_layer_nums - 1):  # -1 because last layer is already applied outside the loop
            out = self.layers[i](out)
            out = self.layernorms[i](out)
            out = torch.relu(out)
            out = self.dropout(out)
        # Element-wise addition of input x and output of function f(x)
        out = x + out
        
        return out
    

class LSTM_Net(nn.Module):
    def __init__(self, hyperparas):
        '''
        hyperparas = {'input_dim':21,'hidden_dim':64,'hidden_nums':1,'output_dim':3,'block_layer_nums':3, 'LSTM_layer_nums':2
        , 'dropout_rate':0.5}
        
        '''
        super(LSTM_Net, self).__init__()
        
        self.input_dim = hyperparas['input_dim']
        self.hidden_dim = hyperparas['hidden_dim']
        self.LSTM_layer_nums = hyperparas['LSTM_layer_nums']
        self.dropout_rate = hyperparas['dropout_rate']
        self.LSTM_layer = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=self.LSTM_layer_nums,
                                  batch_first=True,dropout=self.dropout_rate)
        self.hidden_nums = hyperparas['hidden_nums']
        self.block_layer_nums =hyperparas['block_layer_nums']
        self.output_dim = hyperparas['output_dim']
        # Define layers for the function f (MLP)
        self.layer_list = []

        for _ in range(self.hidden_nums):
            self.layer_list.append(ResNetBlock(hyperparas)
                                   )

        self.layer_list.append(nn.Linear(self.hidden_dim,self.output_dim))

        self.linear_stock = nn.Sequential(*self.layer_list)
        
        
        
    def forward(self, x):
        # Forward pass through the function f (MLP)
        _,(out,c) = self.LSTM_layer(x)
        out = torch.squeeze(out)
        return self.linear_stock(out)