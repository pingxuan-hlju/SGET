import torch.nn.functional as F
from typing import Union
from torch import Tensor
from torch import sum
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch.nn import init
import sys
from early_stopping import EarlyStopping
import random
from torch_geometric.nn import MessagePassing
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
def set_seed():
    seed = 1206
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)

set_seed()
def external_norm(attn):
    softmax = nn.Softmax(dim=0)  # N
    attn = softmax(attn)  # bs,n,S
    attn = attn/sum(attn, dim=2, keepdim=True)  # bs,n,S
    return attn
class DNorm(nn.Module):
    def __init__(
        self,
        dim1=0,dim2=2
    ):
        super().__init__()
        self.dim1=dim1
        self.dim2=dim2
        self.softmax = nn.Softmax(dim=self.dim1)  # N
    def forward(self, attn: Tensor) -> Tensor:
        attn = self.softmax(attn)  # bs,n,S
        return attn
class GEANet(nn.Module):
    def __init__(
            self):
        super().__init__()
        self.dim = 1546
        self.use_shared_unit = True
        self.use_node2_unit = True
        self.unit_size = 256
        self.node1_U1 = nn.Linear( self.dim, self.unit_size)
        self.node1_U2 = nn.Linear(self.unit_size,  256)
        if self.use_node2_unit:
            self.node2_U1 = nn.Linear(self.dim, self.unit_size)
            self.node2_U2 = nn.Linear(self.unit_size, 256)
            if self.use_shared_unit:
                self.share_U = nn.Linear(1546,1546)
                self.norm = DNorm()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self,node_x, embed2):
        if self.use_shared_unit:
            node_x = self.share_U(node_x)
            node2_out = self.share_U(embed2)
            if self.use_node2_unit:
                node2_out = self.node2_U1(node2_out)
                attn = self.norm(node2_out)
                node2_out = self.node2_U2(attn)
        # External attention
        node1_out = self.node1_U1(node_x)
        attn = self.norm(node1_out)
        node1_out = self.node1_U2(attn)
        return node1_out, node2_out,
class GraphConvolutio(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutio, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # Xavier初始化
    def forward(self, adj, features):
        output = torch.matmul(adj, features)
        output = torch.matmul(output, self.weight)
        return output
class GraphConvolutio_2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutio_2, self).__init__()
        self.weight2 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight2)  # Xavier初始化
    def forward(self, adj, features):
        output = torch.matmul(adj, features)
        output = torch.matmul(output, self.weight2)
        return output
class PSVGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PSVGCN, self).__init__()
        self.gc11 = GraphConvolutio(input_dim, hidden_dim)
        self.gc12 = GraphConvolutio(hidden_dim, output_dim)
        self.gc21 = GraphConvolutio_2(input_dim, hidden_dim)
        self.gc22= GraphConvolutio_2(hidden_dim,  output_dim)
    def forward(self, adj, feat1,feat2):
        x1= self.gc11(adj, feat1)
        x2 = self.gc21(adj, feat2)
        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x1 = self.gc12(adj, x1)
        x2 = self.gc22(adj, x2)
        x1 = F.relu(x1)
        x2 = F.relu(x2)
        return x1,x2

class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionMechanism, self).__init__()
        self.feature_dim = feature_dim
        self.weight = nn.Linear(feature_dim, feature_dim)
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, features):
            # 计算注意力分数
        scores = self.leaky_relu(self.weight(features))
            # 归一化
        weights = F.softmax(scores, dim=-1)
        return weights
class FeatureFusion(nn.Module):
    def __init__(self, feat1, feat2):
        super(FeatureFusion, self).__init__()
        self.attention1 = AttentionMechanism(feat1)
        self.attention2 = AttentionMechanism(feat2)
    def forward(self, feat1, feat2):
        alpha = self.attention1(feat1)
        feat1_enhanced = alpha * feat1 + feat1
        beta = self.attention1(feat2)
        feat2_enhanced = beta * feat2 + feat2
        h_fused = feat1_enhanced + feat2_enhanced
        return h_fused

class FaMvf(nn.Module):
    def __init__(self, feat1, feat2):
        super(FaMvf, self).__init__()
        self.feature_fusion = FeatureFusion(feat1, feat2)
    def forward(self, feat1, feat2):
        h_fused = self.feature_fusion(feat1, feat2)
        return h_fused

class TransformerConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads, concat=False, beta=1.0,
                 dropout=0.0, edge_dim=0, bias=True, root_weight=True):
        super(TransformerConv, self).__init__(aggr='add')  # "Add" aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.beta = beta
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.bias = bias
        self.root_weight = root_weight
        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)
        if root_weight:
            self.lin_root = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_edge_h_i = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge_h_j = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge_gate = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        if concat:
            self.out_proj = nn.Linear(heads * out_channels, out_channels, bias=bias)

    def forward(self, x, edge_index, edge_attr):

        query = self.lin_query(x)
        key = self.lin_key(x)
        value = self.lin_value(x)
        # Calculate gate for each edge
        h_i = self.lin_edge_h_i(x)
        h_j = self.lin_edge_h_j(x)
        edge_gate = self.sigmoid(edge_attr + h_i[edge_index[0]] + h_j[edge_index[1]])
        # Perform message passing
        return self.propagate(edge_index, x=x, query=query, key=key,
                              value=value, edge_gate=edge_gate)

    def message(self, query_i, key_j, value_j, edge_gate):
        attention = (query_i * key_j).sum(dim=-1, keepdim=True)  # Dot-product attention
        attention = F.softmax(attention, dim=0)  # Softmax over edges
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        return attention * value_j * edge_gate  # Apply edge gating
    def update(self, aggr_out, x):
        if self.root_weight:
            root = self.lin_root(x)
            aggr_out = aggr_out + root
        if self.concat:
            aggr_out = self.out_proj(aggr_out)
        return aggr_out
class FFMLP(nn.Module):
    def __init__(self, input_dim, ff_dim, dropout):
        super(FFMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, src):
        ss = self.norm1(src)
        ss2 = self.linear2(self.dropout1(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        return self.norm2(ss)
class EGET(nn.Module):
    def __init__(self, in_channels, ff_dim, out_channels, heads, dropout, edge_dim, bias, num_layers, edge_mapping,
                 beta, root_weight):
        super(EGET, self).__init__()
        self.edge_mapping = edge_mapping
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.linear = nn.Linear(128, 1546)
        self.linear2 = nn.Linear(1546, 128)
        self.act = nn.LeakyReLU()
        if edge_mapping:
            self.edge_encoder = nn.Linear(3092, 128)
        else:
            self.edge_encoder = None
        self.convs = nn.ModuleList([
            TransformerConv(in_channels, out_channels, heads, concat=False,
                            beta=beta, dropout=dropout, edge_dim=edge_dim,
                            bias=True, root_weight=root_weight)
            for _ in range(num_layers)
        ])
        self.ffmlps = nn.ModuleList([
            FFMLP(out_channels * heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x, edge_index, edge_attr):
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.ffmlps[i](x)
            x = self.linear(x)
        x = self.act(self.linear2(x))
        return x
# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
class MCMF(nn.Module):
    def __init__(self, feature_dim):
        super(MCMF, self).__init__()
        # 初始化每个模态的激活函数、线性变换和权重参数
        self.w_F = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        # Softmax 函数
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, M_v, M_t):
        # 计算激活特征向量
        M = torch.cat([ M_v, M_t], dim=-1)
        P =  self.leaky_relu(torch.matmul(M,self.w_F) +self.bias )
        # 拼接所有模态的激活特征向量
        # 计算每个模态的权重
        a = self.softmax(P)
        # alpha = nn.Parameter(torch.ones(1)).to(device)
        # 计算加权后的模态特征向量
        F_v = M_v * a[:, 1].unsqueeze(-1)
        F_t = M_t * a[:, 2].unsqueeze(-1)
        # 计算模态互补性
        B_vt = torch.matmul(F_v, F_t.T)
        B_tv = torch.matmul(F_t, F_v.T)
        # 计算互补性权重
        Q_vt = torch.matmul(self.softmax(B_vt), F_t)+ F_v
        Q_tv = torch.matmul(self.softmax(B_tv), F_v)+ F_t
        # 拼接互补特征对
        P_out =torch.cat([Q_vt,Q_tv],dim=1)
        return P_out
class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.c1 = nn.Conv2d(1, 16, kernel_size=(2, 10), stride=1, padding=0)
        self.s1 = nn.MaxPool2d(kernel_size=(1, 10))
        self.c2 = nn.Conv2d(16, 32, kernel_size=(1, 10), stride=1, padding=0)
        self.s2 = nn.MaxPool2d(kernel_size=(1, 10))
        self.leakrelu = nn.LeakyReLU()
        self.mlp = nn.Sequential(nn.Linear(544, 200),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(200, 2),


                                 )
        self.reset_parm()
    # 将线性层，权重初始化
    def reset_parm(self):
        for mode in self.mlp:
            if isinstance(mode, nn.Linear):
                # nn.init.xavier_normal_(mode.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_normal_(mode.weight, gain=nn.init.calculate_gain('leaky_relu'))
    def forward(self, x):
        x = self.s1(self.leakrelu(self.c1(x)))
        x = self.s2(self.leakrelu(self.c2(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x
class final_model(nn.Module):
    def __init__(self):
        super(final_model, self).__init__()
        self.geanet = GEANet()
        self.ga=FaMvf(128,128)
        self.nn=PSVGCN(256,256,128)
        self.GT = EGET(1546, 128, 32, 4, 0.1, 3092, True, 2, True, True, True)
        self.matg=MCMF(256)
        self.cnn = Conv1()
    def forward(self, x1, x2, embeds, embed2,adj,data,data2):
        x3 = x2 + 1373
        xd_original = embeds[x1][:, None, None, :]
        xm_original = embeds[x3][:, None, None, :]
        x_original = torch.cat([xd_original, xm_original],
                               dim=2)
        xd_original2 = embed2[x1][:, None, None, :]
        xm_original2 = embed2[x3][:, None, None, :]
        x_original2 = torch.cat([xd_original2, xm_original2],
                                dim=2)
        x_ea, feat2 = self.geanet(embeds,embed2)
        out1,out2=self.nn(adj,x_ea,feat2)
        G_fused=self.ga(out1,out2)
        x_transformer = self.GT(embeds,data['edge_index'], data['edge_attr'])
        x_transformer2 = self.GT(embed2,data2['edge_index'], data2['edge_attr'])
        T_fused = self.ga( x_transformer, x_transformer2)
        x_matg=self.matg(G_fused,T_fused)
        xdma =  x_matg[x1][:, None, None, :]
        xmma = x_matg[x3][:, None, None, :]
        x_ma = torch.cat([xdma, xmma],
                                        dim=2)
        x = torch.cat([x_original, x_ma],
                      dim=3)
        x = self.cnn(x)
        return x
def train(model, train_set, test_set,feat1, epoch, learn_rate, cross,feat2,adj):
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)
    cost = nn.CrossEntropyLoss()
    embeds = feat1.float().cuda()
    embed2 = feat2.float().cuda()
    adj = adj.float().cuda()#获取邻接矩阵
    #获取边的索引，阈值为 0.8
    threshold = 0.9
    edges = (embeds > threshold).nonzero()  # 返回所有大于 0. 的元素的索引
    edges2 = (embed2 > threshold).nonzero()
    edge_index = edges.t().contiguous()  # 转置为 (2, E) 的格式，E 为边的数量
    edge_index2 = edges2.t().contiguous()
    # 获取节点特征
    node_features = embeds  # 假设节点特征就是原矩阵的值
    node_features2 = embed2
    # 构建边特征
    edge_features = []
    edge_features2 = []
    for i, j in zip(edge_index[0], edge_index[1]):
        node_i_features = node_features[i, :]
        node_j_features = node_features[j, :]
        # 拼接节点 i 和节点 j 的特征
        edge_feature = torch.cat([node_i_features, node_j_features], dim=0)
        edge_features.append(edge_feature)
    for i, j in zip(edge_index2[0], edge_index2[1]):
            node_i_features2 = node_features2[i, :]
            node_j_features2 = node_features2[j, :]
            # 拼接节点 i 和节点 j 的特征
            edge_feature2 = torch.cat([node_i_features2, node_j_features2], dim=0)
            edge_features2.append(edge_feature2)
    # 转换为 tensor
    edge_features = torch.stack(edge_features, dim=0)
    edge_features2 = torch.stack(edge_features2, dim=0)
    # 创建图数据对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    data2 = Data(x=node_features2, edge_index=edge_index2, edge_attr=edge_features2)
    # 打印数据查看
    print(data)
    print(data2)
    early_stopping = EarlyStopping(patience=10, verbose=True, save_path='pt')
    for i in range(epoch):
        model.train()
        LOSS = 0
        for x1, x2, y in train_set:
            x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
            out = model(x1, x2, embeds, embed2, adj,data,data2)
            loss = cost(out, y)
            LOSS += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Cross: %d  Epoch: %d / %d Loss: %0.5f" % (cross + 1, i + 1, epoch, LOSS))
        early_stopping(LOSS, model)
        if early_stopping.early_stop:
            print(f'early_stopping!')
            early_stop = 1
            test(model, test_set, cross, embeds, embed2, adj, data, data2)
            break
        # 如果到最后一轮了，保存测试结果
        if i + 1 == epoch:
            test(model, test_set, cross, embeds, embed2, adj, data, data2)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
def test(model, test_set, cross, embeds, embed2, adj,data,data2):
    correct = 0
    total = 0
    predall, yall = torch.tensor([]), torch.tensor([])
    model.eval()  # 使Dropout失效
    model.load_state_dict(torch.load('pt/best_network3.pth'))
    for x1, x2, y in test_set:
        x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
        with torch.no_grad():
            pred = model(x1, x2, embeds, embed2, adj,data,data2)
        a = torch.max(pred, 1)[1]
        total += y.size(0)
        correct += (a == y).sum()
        predall = torch.cat([predall, torch.as_tensor(pred, device='cpu')], dim=0)
        yall = torch.cat([yall, torch.as_tensor(y, device='cpu')])
    torch.save((predall, yall), './result/fold_%d' % cross)
    print('Test_acc: ' + str((correct / total).item()))
    print("testing finished")
class MyDataset(Dataset):
    def __init__(self, tri, dm):
        self.tri = tri
        self.dm = dm

    def __getitem__(self, idx):
        x, y = self.tri[idx, :]

        label = self.dm[x][y]
        return x, y, label

    def __len__(self):
        return self.tri.shape[0]

if __name__ == "__main__":
    learn_rate = 0.0005
    epoch = 80
    batch = 32
    fea1,fea2,train_xy, test_xy, asso_mat_mask,asso_mat= torch.load('./data_DM/data/embed.pth')
    # fea1, fea2, train_xy, test_xy, asso_mat_mask, asso_mat
    for i in range(5):
        net =final_model().to(device)
        # print(net)
        train_set = DataLoader(MyDataset(train_xy[i], asso_mat), batch, shuffle=True)
        test_set = DataLoader(MyDataset(test_xy[i], asso_mat), batch, shuffle=False)
        train(net, train_set, test_set, fea1[i], epoch, learn_rate, i,fea2[i],asso_mat_mask[i])