import torch
from torch import nn
import torch.nn.functional as F
from collections import Counter
# 对应文中算法1
#z是tensor，包括初始样本和生成样本
def checkId(target, a):
    b = []
    for index, nums in enumerate(a):
        if nums == target:
            b.append(index)
    return (b)

def hierarchical_contrastive_loss(nb_trans, z, labels, alpha=0.34, beta=0.33, temporal_unit=0):
    print("="*50)
    # CBF[8,70,320]
    # 初始化
    loss = torch.tensor(0., device=z.device)
    labels = labels.to(device=z.device)
    d = 0
    while z.size(1) > 1:
        #outer
        if alpha != 0:
            loss += alpha * instance_contrastive_loss_outer(z, labels, nb_trans)
        #inter
        if beta != 0:
            loss += beta * instance_contrastive_loss_inter(z, labels, nb_trans)
        #print(temporal_unit)
        if d >=temporal_unit:
            if 1 - alpha - beta != 0:
                loss += (1 - alpha - beta) * temporal_contrastive_loss(z, nb_trans)
        d += 1
        z = F.max_pool1d(z.transpose(1, 2), kernel_size=2).transpose(1, 2)
        #z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z.size(1) == 1:
        #outer
        if alpha != 0:
            loss += alpha * instance_contrastive_loss_outer(z, labels, nb_trans)
        # inter
        if beta != 0:
            loss += beta * instance_contrastive_loss_inter(z, labels, nb_trans)
        d += 1
    return loss / d
#CBF:z1(8,70,320)   z2(8,70,320)  labels初始输入暂定为一维的tensor
def instance_contrastive_loss_outer(z, labels, nb_trans):
    labels = labels.contiguous().view(-1, 1)#将labels  reshape成为二维张量（24，1）
    logits_mask = torch.eq(labels, labels.T).float()#代表i和j类别相等则为1，反之为0
    logits_labels = torch.tril(logits_mask, diagonal=-1)[:, :-1]
    logits_labels += torch.triu(logits_mask, diagonal=1)[:, 1:]
    B, T = z.size(0)/nb_trans, z.size(1)
    #此时没有损失
    if B == 1:
        return z.new_tensor(0.)
    ############z =   nb_transB x T x C =
    z = z.transpose(0, 1)  # T x nb_transB x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x nb_transB x nb_transB
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x nb_transB x (nb_transB-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:] # T x nb_transB x (nb_transB-1)
    # softmax基础上再加一步log运算(默认以e为底)CBF:70,16,15
    logits = -F.log_softmax(logits, dim=-1)
    logits = logits*logits_labels #CBF:70,16,15
    logits_ave = torch.sum(logits_labels, dim=1)#16
    loss = torch.div(torch.sum(logits, dim=-1), logits_ave).mean() # torch.sum(logits, dim=-1)  CBF: 70,16
    return loss

def instance_contrastive_loss_inter(z, labels, nb_trans):
    B, T = z.size(0)/nb_trans, z.size(1)
    if B == 1:
        return z.new_tensor(0.)
    #z =   nb_transB x T x C
    labels_list = labels
    labels_list = labels_list.tolist()
    #count里面有key和the number of key,其中key和label的值是一样的
    count = Counter(labels_list)
    set_count = set(count)
    class_of_labels = len(set_count)
    if class_of_labels == B:
        return z.new_tensor(0.)
    #loss_sum = torch.tensor(0., device=z.device)
    loss_sum = torch.tensor(0., device=z.device)
    i = 0
    #遍历标签的种类
    for key in count:
        index_label = checkId(key, labels)
        #下面得到的data_key：(x1, x2, x5, x7, x11, x21, x51, x71, x12, x22, x52, x72)
        data_key = z[index_label]
        data_key = data_key.to(device=z.device)
        #重新分配临时标签：每一个初始序列和其同源序列分为一类
        nb_initial = data_key.size(0)/nb_trans#初始序列的数量
        if nb_initial == 1:
            loss_sum += 0
            i +=1
            break
        temperal_label = torch.arange(0, nb_initial)
        temperal_label = temperal_label.to(device=z.device)
        temperal_label = temperal_label.repeat(nb_trans)
        temperal_label = temperal_label.contiguous().view(-1, 1)
        logits_mask = torch.eq(temperal_label, temperal_label.T).float()
        logits_labels = torch.tril(logits_mask, diagonal=-1)[:, :-1]
        logits_labels += torch.triu(logits_mask, diagonal=1)[:, 1:]
        data_key = data_key.transpose(0, 1)  # T x nb_initial*nb_trans x C
        sim = torch.matmul(data_key, data_key.transpose(1, 2))  # T x nb_initial*nb_trans x nb_initial*nb_trans
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x nb_initial*nb_trans x (nb_initial*nb_trans-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]  # T x nb_initial*nb_trans x (nb_initial*nb_trans-1)
        logits = -F.log_softmax(logits, dim=-1)
        logits = logits * logits_labels  # CBF:70,16,15
        logits_ave = torch.sum(logits_labels, dim=1)  # 16
        loss = torch.div(torch.sum(logits, dim=-1), logits_ave).mean()
        loss_sum += loss
        i +=1
    loss_sum = loss_sum/i
    return loss_sum
def temporal_contrastive_loss(z, nb_trans):
    B, T = int(z.size(0)/nb_trans), z.size(1)
    if T == 1:
        return z.new_tensor(0.)
    temperal_label = torch.arange(0, T)
    temperal_label = temperal_label.to(device=z.device)
    temperal_label = temperal_label.repeat(nb_trans)
    temperal_label = temperal_label.contiguous().view(-1, 1)
    logits_mask = torch.eq(temperal_label, temperal_label.T).float()
    logits_labels = torch.tril(logits_mask, diagonal=-1)[:, :-1]
    logits_labels += torch.triu(logits_mask, diagonal=1)[:, 1:]
    #(24,2,320)转成（8,6,320）
    z = z.reshape(B, T*nb_trans, z.size(-1))
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T每一个元素是每一个时间戳和每一个时间戳的表示相乘
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)取下三角https://pytorch.org/docs/master/generated/torch.tril.html
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    #logits_labels = logits_labels.to(device=z.device)
    logits = logits * logits_labels  # CBF:70,16,15
    logits_ave = torch.sum(logits_labels, dim=1)  # 16
    loss = torch.div(torch.sum(logits, dim=-1), logits_ave).mean()  # torch.sum(logits, dim=-1)  CBF: 70,16
    return loss
