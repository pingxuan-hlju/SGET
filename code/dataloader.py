from scipy.io import loadmat
import numpy as np
import random
import torch
import math
import os
from sklearn.model_selection import KFold

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

#  归一化的邻接矩阵
def Regularization(adj):
    row = torch.zeros(1373)
    col = torch.zeros(173)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                row[i] += 1
                col[j] += 1
    row = torch.sqrt(row)
    col = torch.sqrt(col)
    a = torch.Tensor([1])
    ADJ = torch.zeros(size=(1373, 173))
    for m in range(adj.shape[0]):
        for n in range(adj.shape[1]):
            if adj[m][n] == 1:
                temp = row[m] * col[n]
                ADJ[m][n] = torch.div(a, temp)

    return ADJ

def laplace(fea1, fea2):
    G1=fea1
    G2=fea2
    G1=torch.where(G1 < 0.8, torch.zeros_like(G1), G1)
    G2=torch.where(G2 < 0.8, torch.zeros_like(G2), G2)

    dG1=torch.sum(G1,dim=0)**(-1/2)
    dfea1=torch.where(torch.isinf(dG1),torch.full_like(dG1,0),dG1)
    G1=dfea1[:,None]*G1*dfea1[None,:]

    dG2=torch.sum(G2,dim=0)**(-1/2)
    dfea2=torch.where(torch.isinf(dG2),torch.full_like(dG2,0),dG2)
    G2=dfea2[:,None]*G2*dfea2[None,:]
    return G1,G2

# 读入数据
def load_data():
    drug_simi_mat = torch.from_numpy(np.loadtxt("./data_DM/data/drugsimilarity.txt"))
    drug_simi_new=torch.from_numpy(np.loadtxt("./data_DM/data/newheatsimilarity.txt" ))
    micro_simi_mat =torch.from_numpy(np.loadtxt("./data_DM/data/microbe_microbe_similarity.txt"))
    asso_mat = torch.from_numpy(loadmat("./data_DM/data/net1.mat")['interaction'])
    return drug_simi_mat, drug_simi_new, micro_simi_mat, asso_mat
drug_simi_mat, drug_simi_new, micro_simi_mat, asso_mat= load_data()
# @ tensor 版本的shuffle 按维度0
def tensor_shuffle(ts, dim= 0):
    return ts[torch.randperm(ts.shape[dim])]
pos_xy = asso_mat.nonzero()  # 所有正例坐标 torch.Size([2470, 2])
neg_xy = tensor_shuffle((asso_mat == 0).nonzero(), dim=0)  # 所有负例坐标 torch.Size([235059, 2])
rand_num_4940 = torch.randperm(4940) # 2470* 2#随机排列正的索引,生成一个 [0, 4940) 之间的随机索引，用于后续划分数据
neg_xy, rest_neg_xy = neg_xy[0: len(pos_xy)], neg_xy[len(pos_xy):]  # 打乱之后的负例,选取 和正样本数量相等的负样本，剩余的用于测试
print("neg_xy",neg_xy.shape)
print("rest_neg_xy",rest_neg_xy.shape)
pos_neg_xy = torch.cat((pos_xy, neg_xy), dim=0)[rand_num_4940]#合并正负样本,并打乱
kflod = KFold(n_splits=5, shuffle=False)
train_xy = []# 存储 训练集索引
test_xy = []#存储 测试集索引（5 折）
asso_mat_mask =[]#存储 遮掩后的邻接矩阵
asso_mat_mask_1 =[]
fea1 = []#存储 两种不同的嵌入特征
fea2 = []
for fold, (train_xy_idx, test_xy_idx) in enumerate(kflod.split(pos_neg_xy)):
    print(f'第{fold + 1}折')
    train_xy.append(pos_neg_xy[train_xy_idx,])  # 每折的训练集坐标 ，取出当前折的 训练集索引
    test = pos_neg_xy[test_xy_idx]#取出当前折的 测试集索引
    print(" test",  test.shape)
    test_all = torch.cat([test, rest_neg_xy], dim=0)#测试集 = 当前测试集 + 剩余负样本
    print("test_all", test_all.shape)
    test_xy.append(test_all)        # 存储本折的测试集
    print("train_xy",len(train_xy))
    # @ mask test
    asso_mat_zy = asso_mat.clone()#复制 asso_mat，用于 mask 处理。
    for index in test:#将测试集中所有 1 置为 0
        if asso_mat[index[0]][index[1]] == 1:
            asso_mat_zy[index[0]][index[1]] = 0
    O2 = torch.zeros(size=(173, 173))
    O1 = torch.zeros(size=(1373, 1373))
    asso_mat_zy = Regularization(asso_mat_zy)
    row1 = torch.cat([O1, asso_mat_zy], dim=1)
    row2 = torch.cat([asso_mat_zy.T, O2], dim=1)
    double_DM = torch.cat([row1, row2], dim=0)  # 拼接双层邻接矩阵
    asso_mat_mask.append(double_DM)#asso_mat_mask是遮掩后的双层邻接矩阵，1546×1546的
    #asso_mat_zy = Regularization(asso_mat_zy)
    DD_DM = torch.cat([drug_simi_mat, asso_mat_zy], dim=1)
    DDN_DM = torch.cat([drug_simi_new, asso_mat_zy], dim=1)
    DM_MM = torch.cat([asso_mat_zy.T, micro_simi_mat], dim=1)
    embed1 = torch.cat([DD_DM, DM_MM], dim=0)  # 生成embedding -> [1546,1546]
    embed2 = torch.cat([DDN_DM, DM_MM], dim=0)  # 生成embedding -> [1546,1546]
    fea1.append(embed1)
    fea2.append(embed2)
# 1. 保存所有正负样本索引
torch.save(pos_xy, './data_DM/data/pos_xy.pth')
torch.save(neg_xy, './data_DM/data/neg_xy.pth')
torch.save(rest_neg_xy, './data_DM/data/rest_neg_xy.pth')
# 2. 保存每折训练/测试集索引
torch.save(train_xy, './data_DM/data/train_xy.pth')
torch.save(test_xy, './data_DM/data/test_xy.pth')

torch.save([fea1,fea2,train_xy, test_xy,asso_mat_mask,asso_mat],'./data_DM/data/embed1.pth')

#print(train_xy.shape)















