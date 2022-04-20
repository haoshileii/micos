import torch
from torch import nn
import torch.nn.functional as F
from collections import Counter
def checkId(target, a):
    b = []
    for index, nums in enumerate(a):
        if nums == target:
            b.append(index)
    return (b)

def hierarchical_contrastive_loss(nb_trans, z, labels, alpha=0.34, beta=0.33, temporal_unit=0):
    loss = torch.tensor(0., device=z.device)
    labels = labels.to(device=z.device)
    d = 0
    while z.size(1) > 1:
        if alpha != 0:
            loss += alpha * supervised_contrastive_loss_inter(z, labels, nb_trans)
        if beta != 0:
            loss += beta * supervised_contrastive_loss_intra(z, labels, nb_trans)
        if d >=temporal_unit:
            if 1 - alpha - beta != 0:
                loss += (1 - alpha - beta) * self_supervised_contrastive_loss(z, nb_trans)
        d += 1
        z = F.max_pool1d(z.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z.size(1) == 1:
        if alpha != 0:
            loss += alpha * supervised_contrastive_loss_inter(z, labels, nb_trans)
        if beta != 0:
            loss += beta * supervised_contrastive_loss_intra(z, labels, nb_trans)
        d += 1
    return loss / d
def supervised_contrastive_loss_inter(z, labels, nb_trans):
    labels = labels.contiguous().view(-1, 1)
    logits_mask = torch.eq(labels, labels.T).float()
    logits_labels = torch.tril(logits_mask, diagonal=-1)[:, :-1]
    logits_labels += torch.triu(logits_mask, diagonal=1)[:, 1:]
    B, T = z.size(0)/nb_trans, z.size(1)
    if B == 1:
        return z.new_tensor(0.)
    z = z.transpose(0, 1)  # T x nb_transB x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x nb_transB x nb_transB
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x nb_transB x (nb_transB-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:] # T x nb_transB x (nb_transB-1)
    logits = -F.log_softmax(logits, dim=-1)
    logits = logits*logits_labels
    logits_ave = torch.sum(logits_labels, dim=1)
    loss = torch.div(torch.sum(logits, dim=-1), logits_ave).mean()
    return loss

def supervised_contrastive_loss_intra(z, labels, nb_trans):
    B, T = z.size(0)/nb_trans, z.size(1)
    if B == 1:
        return z.new_tensor(0.)
    labels_list = labels
    labels_list = labels_list.tolist()
    count = Counter(labels_list)
    set_count = set(count)
    class_of_labels = len(set_count)
    if class_of_labels == B:
        return z.new_tensor(0.)
    loss_sum = torch.tensor(0., device=z.device)
    i = 0
    for key in count:
        index_label = checkId(key, labels)
        data_key = z[index_label]
        data_key = data_key.to(device=z.device)
        nb_initial = data_key.size(0)/nb_trans
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
        logits = logits * logits_labels
        logits_ave = torch.sum(logits_labels, dim=1)
        loss = torch.div(torch.sum(logits, dim=-1), logits_ave).mean()
        loss_sum += loss
        i +=1
    loss_sum = loss_sum/i
    return loss_sum
def self_supervised_contrastive_loss(z, nb_trans):
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
    z = z.reshape(B, T*nb_trans, z.size(-1))
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    logits = logits * logits_labels
    logits_ave = torch.sum(logits_labels, dim=1)
    loss = torch.div(torch.sum(logits, dim=-1), logits_ave).mean()
    return loss
