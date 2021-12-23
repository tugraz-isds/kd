import torch

def nanmean_tensor(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

# def agg_mean_tensor(samples, labels):
#     pass

# groupby in torch: https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/7

def mean_by_label(samples, labels, n_class = None):
    ''' select mean(samples), count() from samples group by labels order by labels asc '''
    if n_class is None:
        n_class = labels.max()+1

    weight = torch.zeros(n_class, samples.shape[0]).to(samples.device) # L, N
    weight[labels, torch.arange(samples.shape[0])] = 1
    label_count = weight.sum(dim=1)
    weight = torch.nn.functional.normalize(weight, p=1, dim=1) # l1 normalization
    mean = torch.mm(weight, samples) # L, F
    index = torch.arange(mean.shape[0])[label_count > 0]
    return mean, label_count
    # return mean[index], label_count[index]

def sum_by_label_scatter_add(samples, labels, n_class = None):
    # labels = torch.LongTensor([1, 3, 3, 1])
    if n_class is None:
        n_class = labels.max()+1
    labels = torch.LongTensor(labels).to(samples.device)

    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    weight = torch.zeros(n_class, dtype=labels_count.dtype).to(samples.device) # L, N
    weight[unique_labels] = labels_count

    sum_label = torch.zeros((n_class, samples.shape[1]), dtype=samples.dtype, device=samples.device)
    sum_label = sum_label.scatter_add_(0, labels.view(labels.size(0), 1).expand(-1, samples.size(1)), samples)
    return sum_label, weight

# samples = torch.Tensor([
#                      [0.1, 0.1],    #-> group / class 1
#                      [0.2, 0.2],    #-> group / class 2
#                      [0.4, 0.4],    #-> group / class 2
#                      [0.0, 0.0]     #-> group / class 0
#               ])

# labels = torch.LongTensor([1, 3, 3, 1])
# labels2 = torch.LongTensor([0, 2, 2, 0])
# samples2 = samples

# n_class = 5
# n_emb_dim = 2
# m1, w1 = mean_by_label(samples, labels, n_class)
# m2, w2 = mean_by_label(samples2, labels2, n_class)

# (m1*w1.view(-1, 1) + m2*w2.view(-1, 1))/(w1+w2).view(-1,1)


# labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))

# unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

# res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
# res = res / labels_count.float().unsqueeze(1)