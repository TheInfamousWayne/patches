import numpy as np
import torch
import scipy.linalg
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import ipdb


if torch.cuda.is_available():
    device = 'cuda'
    n_gpus = torch.cuda.device_count()
else:
    device = 'cpu'


def select_patches_from_loader(loader, batchsize, patch_size, n_patches, n_images, n_patches_per_rowcol, image_channels=1, func=None, seed=0, stride=1):
    """
    Returns a randomly selected subset of patches of size (n_patches, image_channels, patch_size, patch_size)

    loader=trainloader,
    batchsize=args.batchsize,
    patch_size= the filter dimension, (4 --> 4x4 convolution)
    n_patches=n_channel_convolution, (total number of filters)
    n_images=n_images_trainset, (number of images in the dataset)
    n_patches_per_rowcol=n_patches_per_rowcol, (number of sliding patches along a row)
    func=None, seed=args.numpy_seed, stride=1
    """
    np.random.seed(seed)
    n_patches_per_image = n_patches_per_rowcol**2
    n_patches_total = n_images * n_patches_per_image

    # selecting a subset of random patches from all patches.
    # #All_patches = #Images * #n_patches_per_image
    patch_ids = np.random.choice(n_patches_total, size=n_patches, replace=True)
    unique_patch_ids = np.unique(patch_ids)
    while len(unique_patch_ids) < len(patch_ids):
        unique_patch_ids = np.unique(np.concatenate([unique_patch_ids, np.random.choice(n_patches_total, size=n_patches, replace=True)]))
    patch_ids = unique_patch_ids[:len(patch_ids)]

    # TODO: replace the above with --> patch_ids = np.random.choice(n_patches_total, size=n_patches, replace=False)


    patch_img_batch_ids = (patch_ids % n_images) // batchsize

    selected_patches = torch.DoubleTensor(n_patches, image_channels, patch_size, patch_size).fill_(0)

    for batch_idx, (inputs, _) in enumerate(loader):
        if batch_idx not in patch_img_batch_ids:
            continue

        batch_patch_ids = np.argwhere(patch_img_batch_ids == batch_idx)

        if func is not None:
            inputs = func(inputs)
        inputs = inputs.cpu().double()

        for batch_patch_id in batch_patch_ids:
            patch_id  = patch_ids[batch_patch_id]
            img_id = (patch_id % n_images) % batchsize
            x_id = int(patch_id // n_images % n_patches_per_rowcol)
            y_id = int(patch_id // (n_images * n_patches_per_rowcol))
            selected_patches[batch_patch_id] = inputs[img_id, :, x_id:x_id+patch_size, y_id:y_id+patch_size]

    return selected_patches

def select_patches_randomly(images, patch_size, n_patches=5000000, seed=0, image_channels=1):
    np.random.seed(seed)
    images = images.transpose(0, 3, 1, 2)

    n_patches_per_row = images.shape[2] - patch_size + 1
    n_patches_per_col = images.shape[3] - patch_size + 1
    n_patches_per_image = n_patches_per_row * n_patches_per_col
    n_patches_total = images.shape[0] * n_patches_per_image
    patch_ids = np.random.choice(n_patches_total, size=n_patches, replace=True)
    unique_patch_ids = np.unique(patch_ids)
    while len(unique_patch_ids) < len(patch_ids):
        unique_patch_ids = np.unique(np.concatenate([unique_patch_ids, np.random.choice(n_patches_total, size=n_patches, replace=True)]))
    patch_ids = unique_patch_ids[:len(patch_ids)]

    patches = np.zeros((n_patches, image_channels, patch_size, patch_size), dtype=images.dtype)

    for i_patch, patch_id in enumerate(patch_ids):
        img_id = patch_id % images.shape[0]
        x_id = patch_id // images.shape[0] % n_patches_per_row
        y_id = patch_id // (images.shape[0] * n_patches_per_row)
        patches[i_patch] = images[img_id, :, x_id:x_id+patch_size, y_id:y_id+patch_size]

    return patches


def correct_topk(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum().item()
            res.append(correct_k)

    return res


def compute_whitening_from_loader(loader, patch_size, seed=0, stride=1, func=None):
    """
    ##################################################################
    ##################### Computing Whitening ########################
    ##################################################################

    This part saves the mean and covariance of the whitening operator. the mean and covariance is calculated over the
    whole dataset.
    All the patches from all the images of the dataset are then extracted to calculate the mean and cov.
    Cov matrix is saved in its diagonalised form, i.e. the eigen values and the eigen vectors.

    All returned objects are numpy arrays.

    NOTE: This just saves the mean and sigma matrix. The lambda hyperparameter of whitening is not used here.

    Reference: https://en.wikipedia.org/wiki/Diagonalizable_matrix#Diagonalization

    """

    mean, covariance = None, None

    # compute the mean
    N = 0
    torch.manual_seed(seed)
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs, _ = inputs.to(device), _.to(device)
        if func is not None:
            inputs = func(inputs)
        patches = F.unfold(inputs, patch_size, stride=stride).transpose(0, 1).contiguous()  # extracting patches from the whole image
        patches = patches.view(patches.size(0), -1)
        n = inputs.size(0)
        batch_mean = patches.mean(dim=1, keepdims=True).double()
        if mean is None:
            mean = batch_mean
        else:
            mean = N/(N+n)*mean + n/(N+n)*batch_mean
        N += n

    mean = mean.float()
    # compute the covariance
    N = 0
    torch.manual_seed(seed)
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs, _ = inputs.to(device), _.to(device)
        if func is not None:
            inputs = func(inputs)
        patches = F.unfold(inputs, patch_size, stride=stride).transpose(0, 1).contiguous()
        patches = patches.view(patches.size(0), -1) - mean
        n = inputs.size(0)
        batch_covariance = (patches @ patches.t() / patches.size(1))
        if covariance is None:
            covariance = batch_covariance.double()
        else:
            covariance = N/(N+n)*covariance + n/(N+n)*batch_covariance.double()
        N += n

    (eigvals, eigvecs) = scipy.linalg.eigh(covariance.cpu().numpy())

    mean = mean.view(-1).cpu().numpy().astype('float32')

    return (mean, eigvecs, eigvals)

def compute_whitening(patches, reg=0.001):
    if (patches.dtype == 'uint8'):
        patches = patches.astype('float64')
        patches /= 255.0

    orig_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)

    patches_mean = np.mean(patches, axis=0, keepdims=True)

    patches = patches - patches_mean

    if reg < 0 :
        return (patches.reshape(orig_shape).astype('float32'),
                np.eye(patches.shape[1]).astype('float32'),
                patches_mean.reshape(-1).astype('float32'))

    covariance_matrix = patches.T.dot(patches) / patches.shape[0]

    (eigvals, eigvecs) = scipy.linalg.eigh(covariance_matrix)

    inv_sqrt_eigvals = np.diag(np.power(eigvals + reg, -1/2))

    whitening_operator = eigvecs.dot(inv_sqrt_eigvals)

    return (patches_mean.reshape(-1).astype('float32'),
            whitening_operator.astype('float32'))


def heaviside(x, bias):
    if x.dtype == torch.float16:
        return (x > bias).half()
    return (x > bias).float()


def topk(x, k):
    x_abs = torch.abs(x)
    if x.dtype == torch.float16:
        return (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).half() * x
    return (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).float() * x


def topk_heaviside(x, k):
    if x.dtype == torch.float16:
        # return (x > x.topk(dim=1, k=(k+1)).values.min(dim=1, keepdim=True).values).half()
        # return (x > x.topk(dim=1, k=(k+1)).values[:,-1:,:,:]).half()
        return (x > x.kthvalue(dim=1, k=k-1, keepdim=True).values).half()
    return (x > x.topk(dim=1, k=(k+1)).values.min(dim=1, keepdim=True).values).float()


def compute_classifier_outputs(outputs1, outputs2, targets, args, batch_norm1, batch_norm2, batch_norm, classifier1, classifier2, classifier, train=True, relu_after_bottleneck=False):
    if args.batch_norm:
        outputs1, outputs2 = batch_norm1(outputs1), batch_norm2(outputs2)


    if args.convolutional_classifier == 0:
        outputs1, outputs2 = outputs1.view(outputs1.size(0),-1), outputs2.view(outputs2.size(0),-1)

    outputs1, outputs2 = classifier1(outputs1), classifier2(outputs2)
    outputs = outputs1 + outputs2


    if args.convolutional_classifier > 0:
        if args.bottleneck_dim > 0:
            if args.relu_after_bottleneck:
                outputs = F.relu(outputs)
                if args.dropout > 0 and train:
                    outputs = F.dropout(outputs, p=args.dropout)
                if batch_norm is not None:
                    outputs = batch_norm(outputs)
            outputs = classifier(outputs)
            outputs = F.adaptive_avg_pool2d(outputs, 1)
        else:
            outputs = F.adaptive_avg_pool2d(outputs, 1)
    elif args.bottleneck_dim > 0:
        if args.relu_after_bottleneck:
            outputs = F.relu(outputs)
            if args.dropout > 0:
                outputs = F.dropout(outputs, p=args.dropout)
        outputs = classifier(outputs)

    outputs = outputs.view(outputs.size(0),-1)

    return outputs, targets  # for fdg_uptake_class
    return outputs.float(), targets.float()  # for other metadata


def create_classifier_blocks(out1, out2, args, params, n_classes):
    batch_norm1, batch_norm2, batch_norm, classifier1, classifier2, classifier =  None, None, None, None, None, None

    if args.batch_norm:
        batch_norm1 = nn.BatchNorm2d(out1.size(1), affine=(not args.no_affine_batch_norm)).to(device).float()
        batch_norm2 = nn.BatchNorm2d(out2.size(1), affine=(not args.no_affine_batch_norm)).to(device).float()
        params += list(batch_norm1.parameters()) + list(batch_norm2.parameters())


    if args.convolutional_classifier > 0:
        if args.bottleneck_dim > 0:
            classifier = nn.Conv2d(args.bottleneck_dim, n_classes, args.convolutional_classifier).to(device).float()
            if args.bn_after_bottleneck:
                batch_norm = nn.BatchNorm2d(args.bottleneck_dim, affine=(not args.no_affine_batch_norm)).to(device).float()
                params += list(batch_norm.parameters())
            params += list(classifier.parameters())
            classifier1 = nn.Conv2d(out1.size(1), args.bottleneck_dim, args.bottleneck_spatialsize, stride=args.bottleneck_stride).to(device).float()
            classifier2 = nn.Conv2d(out2.size(1), args.bottleneck_dim, args.bottleneck_spatialsize, stride=args.bottleneck_stride).to(device).float()
        else:
            classifier1 = nn.Conv2d(out1.size(1), n_classes, args.convolutional_classifier).to(device).float()
            classifier2 = nn.Conv2d(out2.size(1), n_classes, args.convolutional_classifier).to(device).float()
    else:
        out1, out2 = out1.view(out1.size(0), -1), out2.view(out1.size(0), -1)
        if args.bottleneck_dim > 0:
            classifier = nn.Linear(args.bottleneck_dim, n_classes).to(device).float()
            params += list(classifier.parameters())
            classifier1 = nn.Linear(out1.size(1), args.bottleneck_dim).to(device).float()
            classifier2 = nn.Linear(out2.size(1), args.bottleneck_dim).to(device).float()
        else:
            classifier1 = nn.Linear(out1.size(1), n_classes).to(device).float()
            classifier2 = nn.Linear(out2.size(1), n_classes).to(device).float()

    params += list(classifier1.parameters()) + list(classifier2.parameters())

    return batch_norm1, batch_norm2, batch_norm, classifier1, classifier2, classifier


def compute_channel_mean_and_std(loader, net, n_channel_convolution,
        kernel_convolution, whitening_operator,
        minus_whitened_patches_mean, n_epochs=1,
        seed=0):

    mean1, mean2 = torch.DoubleTensor(n_channel_convolution).fill_(0).to(device), torch.DoubleTensor(n_channel_convolution).fill_(0).to(device)
    std1, std2 = torch.DoubleTensor(n_channel_convolution).fill_(0).to(device), torch.DoubleTensor(n_channel_convolution).fill_(0).to(device)

    print(' first pass to compute the mean')
    N = 0
    torch.manual_seed(seed)
    for i_epoch in range(n_epochs):
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs, _ = inputs.to(device), _.to(device)

            with torch.no_grad():
                if len(kernel_convolution) > 1:
                    outputs = []
                    for i in range(len(kernel_convolution)):
                        outputs.append(net(inputs, kernel_convolution[i], whitening_operator, minus_whitened_patches_mean))
                    outputs1 = torch.cat([out[0] for out in outputs], dim=1)
                    outputs2 = torch.cat([out[1] for out in outputs], dim=1)
                    del outputs
                else:
                    outputs1, outputs2 = net(inputs, kernel_convolution[0], whitening_operator, minus_whitened_patches_mean)
                n = inputs.size(0)
                mean1 = N/(N+n) * mean1 + outputs1.mean(dim=(0, 2, 3)).double() * n/(N+n)
                mean2 = N/(N+n) * mean2 + outputs2.mean(dim=(0, 2, 3)).double() * n/(N+n)
                N += n

    mean1 = mean1.view(1, -1, 1, 1)
    mean2 = mean2.view(1, -1, 1, 1)
    print(' second pass to compute the std')
    N = 0
    torch.manual_seed(seed)
    for i_epoch in range(n_epochs):
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs, _ = inputs.to(device), _.to(device)

            with torch.no_grad():
                if len(kernel_convolution) > 1:
                    outputs = []
                    for i in range(len(kernel_convolution)):
                        outputs.append(net(inputs, kernel_convolution[i], whitening_operator, minus_whitened_patches_mean))
                    outputs1 = torch.cat([out[0] for out in outputs], dim=1)
                    outputs2 = torch.cat([out[1] for out in outputs], dim=1)
                    del outputs
                else:
                    outputs1, outputs2 = net(inputs, kernel_convolution[0], whitening_operator, minus_whitened_patches_mean)
                n = inputs.size(0)
                std1 = N/(N+n) * std1 + ((outputs1 - mean1)**2).mean(dim=(0, 2, 3)).double() * n/(N+n)
                std2 = N/(N+n) * std2 + ((outputs2 - mean2)**2).mean(dim=(0, 2, 3)).double() * n/(N+n)
                N += n
    std1, std2 = torch.sqrt(std1), torch.sqrt(std2)

    return mean1.float(), mean2.float(), std1.float().view(1, -1, 1, 1), std2.float().view(1, -1, 1, 1)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, k=2, n=4, num_classes=4):
        super(ResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.K = in_channels
        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
            nn.Conv2d(in_channels, self.ichannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU(True)
        )

        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def eval_L_rbf(X, Y=None, sig=5.):
    X = np.atleast_2d(X)
    X_norm_sq = np.linalg.norm(X, axis=1)**2
    if Y is None:
        pairwise_distance_sq = X_norm_sq[:, np.newaxis] - 2*X.dot(X.T) + X_norm_sq[np.newaxis,:]
    else:
        Y = np.atleast_2d(Y)
        Y_norm_sq = np.linalg.norm(Y, axis=1)**2
        pairwise_distance_sq = X_norm_sq[:, np.newaxis] - 2*X.dot(Y.T) + Y_norm_sq[np.newaxis,:]
    return np.exp(-pairwise_distance_sq / sig**2)


def compute_features(loader, net, args, flip=False):
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            if torch.cuda.is_available():
                inputs = inputs.half()
            if args.batchsize_net > 0:
                outputs = []
                for i in range(np.ceil(inputs.size(0)/args.batchsize_net).astype('int')):
                    start, end = i*args.batchsize_net, min((i+1)*args.batchsize_net, inputs.size(0))
                    inputs_batch = inputs[start:end].to(device)
                    outputs.append(net(inputs_batch))
                outputs1 = torch.cat([out[0] for out in outputs], dim=0)
                outputs2 = torch.cat([out[1] for out in outputs], dim=0)
            else:
                inputs = inputs.to(device)
                outputs1, outputs2 = net(inputs)
            outputs1, outputs2 = outputs1.float(), outputs2.float()
            feats = torch.cat([
                    outputs1.reshape((outputs1.size(0), -1)), outputs2.reshape((outputs1.size(0), -1))
                ], dim=1).cpu().numpy()
            features.append(feats)
            labels.append(targets.cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)



def calculate_HRank(outputs1, outputs2, targets):

    # get ranks of feature maps
    batch_item_idx_for_label = {}
    avg_batch_rank_per_label = {}

    unique_classes = [i.item() for i in targets.unique()]
    for c in unique_classes:
        batch_item_idx_for_label[c] = (targets == c).nonzero(as_tuple=True)[0]

    R1 = torch.linalg.matrix_rank(outputs1.float())
    R2 = torch.linalg.matrix_rank(outputs2.float())

    avg_batch_rank_per_label = {
        c: [R1[batch_item_idx_for_label[c]].float().mean(0),
            R2[batch_item_idx_for_label[c]].float().mean(0)]
        for c in unique_classes}

    return avg_batch_rank_per_label


def save_rank_statistics(feature_maps_rank_per_batch, net):
    # plot subplot of feature rank statistics for 4 classes
    plt.clf()
    plt.cla()
    pos_filters, neg_filters = {}, {}
    unique_classes = feature_maps_rank_per_batch[0].keys()
    for c in unique_classes:
        rank_for_label = [i[c] for i in feature_maps_rank_per_batch]
        p, n = zip(*rank_for_label)
        p = np.array([i.cpu().detach().numpy() for i in p])
        n = np.array([i.cpu().detach().numpy() for i in n])
        pos_filters[c] = p
        neg_filters[c] = n

    sorted_pos_idx_label0 = np.argsort(pos_filters[0].mean(axis=0))
    sorted_neg_idx_label0 = np.argsort(neg_filters[0].mean(axis=0))

    # rank statistic for positive filters
    N = len(unique_classes)
    fig1, ax1 = plt.subplots(2, 2)
    for idx, c in enumerate(unique_classes):
        ax1[idx//2, idx%2].imshow(pos_filters[c], interpolation='nearest', aspect='auto')

    plt.savefig(f"./figures/pos_filter_rank_statistic.png")

    # rank statistic for negative filters
    plt.clf()
    plt.cla()
    fig2, ax2 = plt.subplots(2, 2)
    for c in unique_classes:
        ax2[c//2, c%2].imshow(neg_filters[c], interpolation='nearest', aspect='auto')

    plt.savefig(f"./figures/neg_filter_rank_statistic.png")

    # difference between label 3 and label 0 activations
    plt.clf()
    plt.cla()
    fig3, ax3 = plt.subplots(1, 2)

    pos_difference = pos_filters[3] - pos_filters[0]
    neg_difference = neg_filters[3] - neg_filters[0]

    sorted_pos_idx = np.argsort(pos_difference.mean(axis=0))
    sorted_neg_idx = np.argsort(neg_difference.mean(axis=0))

    sort_flag = True
    if sort_flag:
        pos_difference = pos_difference[:, sorted_pos_idx]
        neg_difference = neg_difference[:, sorted_neg_idx]

    im0 = ax3[0].imshow(pos_difference, interpolation='nearest', aspect='auto', cmap='viridis')
    plt.colorbar(im0, ax=ax3[0])

    im1 = ax3[1].imshow(neg_difference, interpolation='nearest', aspect='auto', cmap='viridis')
    plt.colorbar(im1, ax=ax3[1])

    plt.savefig(f"./figures/difference_rank_statistic.png")

    # show filters active in passive regions and active regions
    pos_filters_for_active_regions = sorted_pos_idx[pos_difference.mean(0) > 3.2]
    pos_filters_for_passive_regions = sorted_pos_idx[pos_difference.mean(0) < -1]

    neg_filters_for_active_regions = sorted_neg_idx[neg_difference.mean(0) > 3.2]
    neg_filters_for_passive_regions = sorted_neg_idx[neg_difference.mean(0) < -2.5]

    # display random 25 for each combination
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(2, 2)

    pos_filters_for_active_regions = net.kernel_convolution[np.random.choice(pos_filters_for_active_regions, 25)]
    pos_filters_for_passive_regions = net.kernel_convolution[np.random.choice(pos_filters_for_passive_regions, 25)]
    neg_filters_for_active_regions = net.kernel_convolution[np.random.choice(neg_filters_for_active_regions, 25)]
    neg_filters_for_passive_regions = net.kernel_convolution[np.random.choice(neg_filters_for_passive_regions, 25)]

    all_filters = [pos_filters_for_active_regions,
                   pos_filters_for_passive_regions,
                   neg_filters_for_active_regions,
                   neg_filters_for_passive_regions]

    def get_concatenated_filter(f):
        data = np.ones((24, 24))
        data *= min(f.cpu().numpy().reshape(-1))
        f = torch.squeeze(f)
        for i in range(25):
            # data *= min(f[i].cpu().numpy().reshape(-1))
            data[i//5*4 + i//5: i//5*4 + 4 + i//5,
                 i%5*4 + i%5: i%5*4 + 4 + i%5] = f[i].cpu()
        return data

    for idx, fil in enumerate(all_filters):
        concatenated_filter = get_concatenated_filter(fil)
        ax[idx//2, idx%2].imshow(concatenated_filter, interpolation='nearest', aspect='auto')

    plt.savefig("./figures/sample_filters.png")






#%%
class FileObject(object):
    def __init__(
        self,
        config_file={"split_parts": {},
                     "use_file": {"file_part": 2},
                     "plot_fdg_uptake": {},
                     "remove_lipids": {},
                     "imputation": {"name": "imputation_1"},
                    },
        dir_path=Path(os.getenv("OMICS_DATA")),
        file_name="ERC15_Metabolomi_data.csv",
    ):
        self.df = pd.read_csv(dir_path / file_name, header=None)
        self.files: list
        self.file: pd.DataFrame
        if config_file:
            self.run_from_config(config_file)

    def split_parts(self):
        def set_row_as_column_header(df, row=0):
            df.columns = df.iloc[row]
            df.drop(df.index[row], inplace=True)
            df.set_index(df.columns[0], inplace=True)
            return df

        self.files = [
            set_row_as_column_header(self.df[:7], 0),
            set_row_as_column_header(self.df[7:], 0),
        ]

    def use_file(self, file_part=2):
        self.file = self.files[file_part - 1]

    def plot_fdg_uptake(self):
        fig, ax = plt.subplots(1, 2)
        ax[0].hist(sorted(self.files[0].iloc[2][1:].astype(float)))
        ax[1].hist(sorted(self.files[0].iloc[3][1:].astype(int)), bins=4)

    def remove_lipids(self, update_all_versions=False):
        if update_all_versions:
            self.files[1] = self.files[1][
                self.files[1]["C: Metabo.Class"] != "acylcarnitines"
            ]
            self.files[1] = self.files[1][
                self.files[1]["C: Metabo.Class"] != "glycerophospholipids"
            ]
            self.files[1] = self.files[1][
                self.files[1]["C: Metabo.Class"] != "sphingolipids"
            ]
            self.file = self.files[1]
        else:
            self.file = self.file[self.file["C: Metabo.Class"] != "acylcarnitines"]
            self.file = self.file[
                self.file["C: Metabo.Class"] != "glycerophospholipids"
            ]
            self.file = self.file[self.file["C: Metabo.Class"] != "sphingolipids"]

    def imputation_1(self):
        """
        basic imputation by
            1. dropping rows with all zeros
            2. replacing 0 with column_min/2
        :param type: the type of imputation
        :return:
        """
        self.file.drop(columns=["C: Metabo.Class"], inplace=True)
        self.file = self.file.apply(pd.to_numeric)
        self.file = self.file[(self.file.T != 0).any()]  # dropping rows with all zeros
        imputation_dict = {}
        for c in self.file.columns:  # iterating through the columns
            min_value = (
                self.file[c].drop_duplicates().nsmallest(2).iloc[-1]
            )  # column min (after 0)
            imputation_dict[c] = {0: min_value / 2}  # replace 0 with min_value/2

        self.file = self.file.replace(imputation_dict)

    def imputaiton_2(self):
        print("second")

    def imputation(self, name: int, **kwargs):
        func = getattr(self, name)
        func(**kwargs)

    def run_from_config(self, config_file):
        for func_name, kwargs in config_file.items():
            func = getattr(self, func_name)
            func(**kwargs)

    def decompose_data(self, algorithm, **kwds):
        self.data = self.file.T.to_numpy()
        self.data = algorithm(**kwds).fit_transform(self.data)

    def scatter_plot(self, dimensions=None, uptake=False, title=""):
        plot_kwds = {"alpha": 0.25, "s": 80, "linewidths": 0}
        if dimensions is None:
            dimensions = [0, 1]
        if not hasattr(self, "classes"):
            self.fdg_classes()
        labels = self.classes
        if uptake and not hasattr(self, "uptake"):
            self.fdg_uptake()

        palette = sns.color_palette("deep", np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        fig, ax = plt.subplots(1, 1)
        ax.scatter(
            self.data.T[dimensions[0]],
            self.data.T[dimensions[1]],
            c=colors,
            **plot_kwds
        )
        for i, l in enumerate(labels):
            ax.annotate(
                l,  # this is the text
                (
                    self.data[i][dimensions[0]],
                    self.data[i][dimensions[1]],
                ),  # this is the point to label
                textcoords="offset points",  # how to position the text
                xytext=(0, 5),  # distance from text to points (x,y)
                ha="center",  # horizontal alignment can be left, right or center
                fontsize=5,
            )

            if uptake:
                ax.annotate(
                    self.uptake[i],  # this is the text
                    (
                        self.data[i][dimensions[0]],
                        self.data[i][dimensions[1]],
                    ),  # this is the point to label
                    textcoords="offset points",  # how to position the text
                    xytext=(0, -2),  # distance from text to points (x,y)
                    ha="center",  # horizontal alignment can be left, right or center
                    fontsize=5,
                )

        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title(title, fontsize=24)

    def fdg_uptake(self):
        mouse_region_to_fdg = self.files[0].iloc[4].to_dict()
        mouse_region_label = list(self.file.T.index)  # mouse/region list
        # for exact FDG values
        mouse_region_to_uptake = self.files[0].iloc[3].to_dict()
        self.uptake = list(map(mouse_region_to_uptake.get, mouse_region_label))

    def fdg_classes(self):
        # MOUSE/REGION to class (0,1,2,3) dict
        mouse_region_to_fdg = self.files[0].iloc[4].to_dict()
        mouse_region_label = list(self.file.T.index)  # mouse/region list
        classes = list(
            map(mouse_region_to_fdg.get, mouse_region_label)
        )  # mouse/region --> FDG classes
        # unique_classes = list(set(classes))
        unique_classes = ["0--5", "5--10", "10--15", "15--20"]
        modified_class_dict = {k: v for v, k in enumerate(unique_classes)}
        classes = list(
            map(modified_class_dict.get, classes)
        )  # FDG uptake classification
        self.classes = classes


