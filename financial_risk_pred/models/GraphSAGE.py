import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(
        self, 
        in_c: int, 
        h_c: int or list,  # 修改类型注释，支持 list
        out_c: int, 
        num_layers: int = 2,      
        dropout: float = 0.05, 
        aggr: str = 'mean'        
    ):
        super(GraphSAGE, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

        # -------------------------------------------------------
        # 1. 解析隐藏层参数：统一转为列表
        # -------------------------------------------------------
        # 如果传入的是整数 (e.g., 256)，则将其复制 (num_layers - 1) 次
        # 如果传入的是列表 (e.g., [256, 128, 64])，则直接使用
        if isinstance(h_c, int):
            hidden_dims = [h_c] * (num_layers - 1)
        else:
            hidden_dims = h_c
        
        # 简单校验：列表长度是否足够
        # 如果 num_layers=3，我们需要 2 个隐藏层维度 (Input->H1, H1->H2, H2->Output)
        # 所以列表长度至少要是 num_layers - 1
        if len(hidden_dims) < num_layers - 1:
            raise ValueError(f"h_c list length ({len(hidden_dims)}) must be at least num_layers - 1 ({num_layers - 1})")
            
        # -------------------------------------------------------
        # 2. 动态构建网络层
        # -------------------------------------------------------
        
        # --- 第 1 层 (Input -> Hidden_1) ---
        # 输入维度是 in_c，输出维度是 hidden_dims[0]
        self.convs.append(SAGEConv(in_c, hidden_dims[0], aggr=aggr))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dims[0]))

        # --- 中间层 (Hidden_i -> Hidden_i+1) ---
        # 从 hidden_dims[0] 到 hidden_dims[1], ...
        for i in range(num_layers - 2):
            current_dim = hidden_dims[i]
            next_dim = hidden_dims[i+1]
            
            self.convs.append(SAGEConv(current_dim, next_dim, aggr=aggr))
            self.bns.append(torch.nn.BatchNorm1d(next_dim))

        # --- 最后 1 层 (Hidden_last -> Output) ---
        # 输入是最后一个隐藏层的维度，输出是分类数 out_c
        # 注意：这里取的是 num_layers-2 对应的维度，即 hidden_dims[num_layers-2]
        last_hidden_dim = hidden_dims[num_layers - 2]
        self.convs.append(SAGEConv(last_hidden_dim, out_c, aggr=aggr))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj, **kwargs):
        # 遍历除了最后一层之外的所有层
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, adj)
            x = self.bns[i](x)        
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层
        x = self.convs[-1](x, adj)
        
        return F.log_softmax(x, dim=-1)