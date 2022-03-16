import ipdb
import argparse
import ast
import hashlib
import json
import numpy as np
import os
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# For datasets
from torchvision.datasets import CIFAR10
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dtd import Dataloder

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from imagenet import Imagenet32
import utils

from pathlib import Path
import csv


print('kneighbors.py')
parser = argparse.ArgumentParser(
    'linear classification using patches k nearest neighbors indicators for euclidean metric')

# parameters for the dataset
parser.add_argument('--skip_mouse_id', help="the mouse data to skip", default='-1', type=str)
parser.add_argument('--task_name', help="the training task", default='fdg_uptake_class', type=str)

# parameters for the patches
parser.add_argument('--dataset', help="cifar10/?", default='cifar10')
parser.add_argument('--no_padding', action='store_true', help='no padding used')
parser.add_argument('--patches_file', help=".t7 file containing patches", default='')
parser.add_argument('--n_channel_convolution', default=128, type=int)  # number of patches (or filters)
parser.add_argument('--spatialsize_convolution', default=4, type=int)  # filter dimension
parser.add_argument('--padding_mode', default='constant', choices=['constant', 'reflect', 'symmetric'],
                    help='type of padding for torch RandomCrop')
parser.add_argument('--whitening_reg', default=0.001, type=float,
                    help='regularization bias for zca whitening, negative values means no whitening')
parser.add_argument('--gaussian_patches', action='store_true', help='patches sampled for gaussian RV')
parser.add_argument('--learn_patches', action='store_true', help='learn the patches by SGD')
parser.add_argument('--input_channels', default=1, type=int, help='number of channels in the input image')

# parameters for the second layer of patches
parser.add_argument('--n_channel_convolution_2', default=0, type=int)
parser.add_argument('--spatialsize_convolution_2', default=0, type=int)
parser.add_argument('--whitening_reg_2', default=1e-3, type=float,
                    help='regularization bias for second zca whitening, negative values means no whitening')
parser.add_argument('--kneighbors_2', default=0, type=int)
parser.add_argument('--kneighbors_fraction_2', default=0.25, type=float)
parser.add_argument('--sigmoid_2', default=0., type=float)

# parameters for the extraction
parser.add_argument('--stride_convolution', default=1, type=int)
parser.add_argument('--stride_avg_pooling', default=2, type=int)
parser.add_argument('--spatialsize_avg_pooling', default=3, type=int)
parser.add_argument('--kneighbors', default=0, type=int)
parser.add_argument('--kneighbors_fraction', default=0.25, type=float)
parser.add_argument('--finalsize_avg_pooling', default=0, type=int)
parser.add_argument('--sigmoid', default=0., type=float)
parser.add_argument('--dpp_subsample', action='store_true', help='subsample patches with DPP')

# parameters of the classifier
parser.add_argument('--batch_norm', action='store_true', help='add batchnorm before classifier')
parser.add_argument('--no_affine_batch_norm', action='store_true', help='affine=False in batch norms')
parser.add_argument('--normalize_net_outputs', action='store_true',
                    help='precompute the mean and std of the outputs to normalize them (alternative to batch norm)')
parser.add_argument('--bottleneck_dim', default=0, type=int, help='bottleneck dimension for the classifier')
parser.add_argument('--convolutional_classifier', type=int, default=0,
                    help='size of the convolution for convolutional classifier')
parser.add_argument('--bottleneck_spatialsize', type=int, default=1, help='spatial size of the bottleneck')
parser.add_argument('--bottleneck_stride', type=int, default=1, help='spatial size of the bottleneck')
parser.add_argument('--relu_after_bottleneck', action='store_true', help='add relu after bottleneck ')
parser.add_argument('--bn_after_bottleneck', action='store_true', help='add batch norm after bottleneck ')
parser.add_argument('--dropout', type=float, default=0., help='dropout after relu')
parser.add_argument('--feat_square', action='store_true', help='add square features')
parser.add_argument('--resnet', action='store_true', help='resnet classifier')
parser.add_argument('--loss_type', default='cross_entropy_loss', help='classifier/regeressor loss function')

# parameters of the optimizer
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--batchsize_net', type=int, default=0)
parser.add_argument('--lr_schedule', type=str, default='{0:1e-3, 1:1e-4}')
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam')
parser.add_argument('--sgd_momentum', type=float, default=0.)
parser.add_argument('--weight_decay', type=float, default=0.)

# hardware parameters
parser.add_argument('--path_train', help="path to imagenet", default=str(Path(os.getenv('PATH_TRAIN'))))
parser.add_argument('--path_test', help="path to imagenet", default='/d1/dataset/imagenet32/out_data_val')
parser.add_argument('--path', help="path to imagenet", default='/d1/dataset/2012')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--multigpu', action='store_true')
parser.add_argument('--no_cudnn', action='store_true', help='disable cuDNN to prevent cuDNN error (slower)')
parser.add_argument('--no_jit', action='store_true', help='disable torch.jit optimization to prevent error (slower)')
parser.add_argument('--verbose', action='store_true', help='print training epoch loss per batch')

# reproducibility parameters
parser.add_argument('--numpy_seed', type=int, default=0)
parser.add_argument('--torch_seed', type=int, default=0)
parser.add_argument('--save_model', action='store_true', help='saves the model')
parser.add_argument('--save_best_model', action='store_true', help='saves the best model')
parser.add_argument('--resume', default='', help='filepath of checkpoint to load the model')
parser.add_argument('--summary_file', default=str(Path(os.getenv('SUMMARY_FILE'))), help='file to write summary')

args = parser.parse_args()

if args.batchsize_net > 0:
    assert args.batchsize // args.batchsize_net == args.batchsize / args.batchsize_net, 'batchsize_net must divide batchsize'

print(f'Arguments : {args}')

learning_rates = ast.literal_eval(args.lr_schedule)

# Extract the parameters
n_channel_convolution = args.n_channel_convolution
stride_convolution = args.stride_convolution
spatialsize_convolution = args.spatialsize_convolution
stride_avg_pooling = args.stride_avg_pooling
spatialsize_avg_pooling = args.spatialsize_avg_pooling
finalsize_avg_pooling = args.finalsize_avg_pooling
n_gpus = 1
if torch.cuda.is_available():
    device = 'cuda'
    n_gpus = torch.cuda.device_count()
else:
    device = 'cpu'
print(f'device: {device}')
torch.manual_seed(args.torch_seed)
np.random.seed(args.numpy_seed)

train_sampler = None

# Define the dataset
if args.dataset == 'cifar10':
    spatial_size = 32
    padding = 0 if args.no_padding else 4
    transform_train = transforms.Compose([
        transforms.RandomCrop(spatial_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                                              num_workers=args.num_workers)
    n_classes = 10

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False,
                                             num_workers=args.num_workers)
elif args.dataset in ['imagenet32', 'imagenet64', 'imagenet128']:

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    n_arrays_train = 10
    padding = 4
    spatial_size = 32
    if args.dataset == 'imagenet64':
        spatial_size = 64
        padding = 8
    if args.dataset == 'imagenet128':
        spatial_size = 128
        padding = 16
        n_arrays_train = 100
    n_classes = 1000

    if args.no_padding:
        padding = 0

    transforms_train = [
        transforms.RandomCrop(spatial_size, padding=padding, padding_mode=args.padding_mode),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    transforms_test = [transforms.ToTensor(), normalize]

    trainset = Imagenet32(args.path_train, transform=transforms.Compose(transforms_train), sz=spatial_size,
                          n_arrays=n_arrays_train)
    testset = Imagenet32(args.path_test, transform=transforms.Compose(transforms_test), sz=spatial_size)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batchsize, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    n_classes = 1000

# WIP
elif args.dataset in ['imagenet']:
    spatial_size = 64
    traindir = os.path.join(args.path, 'train')
    valdir = os.path.join(args.path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # MODIF
            # transforms.RandomResizedCrop(64),
            transforms.Resize(72),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True,
        num_workers=8, pin_memory=True)
    testset = datasets.ImageFolder(valdir, transforms.Compose([
        # MODIF
        # transforms.Resize(64),
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ]))
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batchsize, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    n_classes = 1000
elif args.dataset == 'DTD':
    spatial_size = 20
    classes, trainset, testset, trainloader, testloader, trainloader_norandom = Dataloder(args.path_train,
                                                                                          spatial_size=spatial_size,
                                                                                          batchsize=args.batchsize,
                                                                                          skip_mouse_id=args.skip_mouse_id,
                                                                                          task_name=args.task_name).getloader()
    n_classes = len(classes)


def lowestk_heaviside(x, k):
    if x.dtype == torch.float16:
        return (x < x.kthvalue(dim=1, k=k + 1, keepdim=True).values).half()
    return (x < x.kthvalue(dim=1, k=k + 1, keepdim=True).values).float()


def lowestk_sigmoid(x, k, sigmoid):
    if x.dtype == torch.float16:
        return torch.sigmoid((x.kthvalue(dim=1, k=k + 1, keepdim=True).values - x) / sigmoid).half()
    return torch.sigmoid((x.kthvalue(dim=1, k=k + 1, keepdim=True).values - x) / sigmoid).float()


def compute_channel_mean_and_std(loader, net, n_channel_convolution,
                                 kernel_convolution, bias_convolution, n_epochs=1, seed=0):
    mean1, mean2 = torch.DoubleTensor(n_channel_convolution).fill_(0).to(device), torch.DoubleTensor(
        n_channel_convolution).fill_(0).to(device)
    std1, std2 = torch.DoubleTensor(n_channel_convolution).fill_(0).to(device), torch.DoubleTensor(
        n_channel_convolution).fill_(0).to(device)

    print('First pass to compute the mean')
    N = 0
    torch.manual_seed(seed)
    with torch.no_grad():
        for i_epoch in range(n_epochs):
            for batch_idx, (inputs, _) in enumerate(loader):
                if torch.cuda.is_available():
                    inputs = inputs.half()  # converting to 16 bit float
                if args.batchsize_net > 0:
                    outputs = []
                    for i in range(np.ceil(inputs.size(0) / args.batchsize_net).astype('int')):
                        start, end = i * args.batchsize_net, min((i + 1) * args.batchsize_net, inputs.size(0))
                        inputs_batch = inputs[start:end].to(device)
                        outputs.append(net(inputs_batch))
                    outputs1 = torch.cat([out[0] for out in outputs], dim=0)
                    outputs2 = torch.cat([out[1] for out in outputs], dim=0)
                else:
                    inputs = inputs.to(device)
                    outputs1, outputs2 = net(inputs)
                outputs1, outputs2 = outputs1.float(), outputs2.float()
                n = inputs.size(0)
                mean1 = N / (N + n) * mean1 + outputs1.mean(dim=(0, 2, 3)).double() * n / (N + n)
                mean2 = N / (N + n) * mean2 + outputs2.mean(dim=(0, 2, 3)).double() * n / (N + n)
                N += n

    mean1 = mean1.view(1, -1, 1, 1).float()
    mean2 = mean2.view(1, -1, 1, 1).float()
    print('Second pass to compute the std')
    N = 0
    torch.manual_seed(seed)
    with torch.no_grad():
        for i_epoch in range(n_epochs):
            for batch_idx, (inputs, _) in enumerate(loader):
                if torch.cuda.is_available():
                    inputs = inputs.half()
                if args.batchsize_net > 0:
                    outputs = []
                    for i in range(np.ceil(inputs.size(0) / args.batchsize_net).astype('int')):
                        start, end = i * args.batchsize_net, min((i + 1) * args.batchsize_net, inputs.size(0))
                        inputs_batch = inputs[start:end].to(device)
                        outputs.append(net(inputs_batch))
                    outputs1 = torch.cat([out[0] for out in outputs], dim=0)
                    outputs2 = torch.cat([out[1] for out in outputs], dim=0)
                else:
                    inputs = inputs.to(device)
                    outputs1, outputs2 = net(inputs)
                outputs1, outputs2 = outputs1.float(), outputs2.float()
                n = inputs.size(0)
                std1 = N / (N + n) * std1 + ((outputs1 - mean1) ** 2).mean(dim=(0, 2, 3)).double() * n / (N + n)
                std2 = N / (N + n) * std2 + ((outputs2 - mean2) ** 2).mean(dim=(0, 2, 3)).double() * n / (N + n)
                N += n

    std1, std2 = torch.sqrt(std1), torch.sqrt(std2)

    return mean1, mean2, std1.float().view(1, -1, 1, 1), std2.float().view(1, -1, 1, 1)


class Net(nn.Module):
    def __init__(self, kernel_convolution, bias_convolution, spatialsize_avg_pooling, stride_avg_pooling,
                 finalsize_avg_pooling, k_neighbors=1, sigmoid=0.):
        super(Net, self).__init__()
        self.kernel_convolution = nn.Parameter(kernel_convolution, requires_grad=False)
        self.bias_convolution = nn.Parameter(bias_convolution, requires_grad=False)
        self.pool_size = spatialsize_avg_pooling
        self.pool_stride = stride_avg_pooling
        self.finalsize_avg_pooling = finalsize_avg_pooling
        self.k_neighbors = k_neighbors
        self.sigmoid = sigmoid

    def forward(self, x):
        out = F.conv2d(x, self.kernel_convolution)

        if self.sigmoid > 0:
            out1 = lowestk_sigmoid(-out + self.bias_convolution, self.k_neighbors, self.sigmoid)
        else:
            out1 = lowestk_heaviside(-out + self.bias_convolution, self.k_neighbors)
        out1 = F.avg_pool2d(out1, self.pool_size, stride=self.pool_stride, ceil_mode=True)
        if self.finalsize_avg_pooling > 0:
            out1 = F.adaptive_avg_pool2d(out1, self.finalsize_avg_pooling)
        if self.sigmoid > 0:
            out2 = lowestk_sigmoid(out + self.bias_convolution, self.k_neighbors, self.sigmoid)
        else:
            out2 = lowestk_heaviside(out + self.bias_convolution, self.k_neighbors)
        out2 = F.avg_pool2d(out2, self.pool_size, stride=self.pool_stride, ceil_mode=True)
        if self.finalsize_avg_pooling > 0:
            out2 = F.adaptive_avg_pool2d(out2, self.finalsize_avg_pooling)
        return out1, out2  # negative and positive convolutions so that the model is sign invariant


"""
##################################################################
##################### Computing Whitening ########################
##################################################################

This part saves the mean and covariance of the whitening operator. the mean and covariance is calculated over the whole dataset.
We extract patches from all the images of the dataset to calculate the mean and cov.
Cov matrix is saved in its diagonalised form, i.e. the eigen values and the eigen vectors.

All returned objects are numpy arrays.

NOTE: This just saves the mean and sigma matrix. The lambda hyperparameter of whitening is not used/modified/calculated here.

Reference: https://en.wikipedia.org/wiki/Diagonalizable_matrix#Diagonalization

"""

# new version, whitening computed on all the patches of the dataset
whitening_file = f'data/whitening_{args.dataset}_patchsize{spatialsize_convolution}.npz'

if not os.path.exists(whitening_file):

    print('Computing whitening...')

    if args.dataset == 'cifar10':
        trainset_whitening = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        trainloader_whitening = torch.utils.data.DataLoader(trainset_whitening, batch_size=1000, shuffle=False,
                                                            num_workers=args.num_workers)
        stride = 1

    elif args.dataset in ['imagenet32', 'imagenet64', 'imagenet128']:
        stride = 2
        trainset_whitening = Imagenet32(args.path_train, transform=transforms.ToTensor(), sz=spatial_size,
                                        n_arrays=n_arrays_train)
        trainloader_whitening = torch.utils.data.DataLoader(
            trainset_whitening, batch_size=100, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

    elif args.dataset in ['imagenet']:
        stride = 2
        spatial_size = 64
        traindir = os.path.join(args.path, 'train')
        valdir = os.path.join(args.path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        trainset_whitening = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]))
        trainloader_whitening = torch.utils.data.DataLoader(
            trainset_whitening, batch_size=100, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

    elif args.dataset == 'DTD':
        trainset_whitening = None  # may take trainset_norandom from Dataloder.getloader()
        trainloader_whitening = trainloader_norandom
        stride = 1

    patches_mean, whitening_eigvecs, whitening_eigvals = utils.compute_whitening_from_loader(trainloader_whitening,
                                                                                             patch_size=spatialsize_convolution,
                                                                                             stride=stride)
    # patch_size (or kernel size) = 4  (i.e. 4x4), as an example.
    # batch_size = 100 for example
    # input_images = (100, 1, 20, 20) for example.
    # patches_mean = shape(product of kernel dim, batch_size * product of image dimension after convolution) = (4x4, 100 * 17x17)
    # eigvecs = shape(square matrix of dimension) = (16,16). each column vector is a "right" eigen vector of the covariance matrix
    # eigvals = (16,) array with elements being the eigen val (in ascending order) of the corresponding eigen vector

    del trainloader_whitening
    del trainset_whitening

    # all objects are numpy arrays
    np.savez(whitening_file, patches_mean=patches_mean,
             whitening_eigvecs=whitening_eigvecs,
             whitening_eigvals=whitening_eigvals)

    print(f'Whitening computed and saved in file {whitening_file}')


"""
##########################################################################################
##################### Loading the Whitening Operator's parameters ########################
##########################################################################################

The mean and sigma matrix (in the form of eigen vals and eigen vecs) are loaded. 

The whitening regularized (lambda) is used here to create the whitening operator:

W = (λI + ∑)^(-1/2)

The operator is applied on the centered patch. (i.e. after subtracting the mean from it)

"""

print(f'Loading whitening from file {whitening_file}...')
whitening = np.load(whitening_file)
whitening_eigvecs = whitening['whitening_eigvecs']  # (16x16) for a patch_size of 4
whitening_eigvals = whitening['whitening_eigvals']  # 16
patches_mean = whitening['patches_mean']  # (16,)

if args.whitening_reg >= 0:
    # inv_sqrt_eigvals = np.diag(np.power(whitening_eigvals + args.whitening_reg, -1/2))
    inv_sqrt_eigvals = np.diag(1. / np.sqrt(whitening_eigvals + args.whitening_reg))
    whitening_op = whitening_eigvecs.dot(inv_sqrt_eigvals).astype('float32')  # P @ Λ^(-1/2)
    # P * Λ^(-1/2),  P is a row matrix with each element being a column eigen vector.

else:  # without the lambda regularisation
    whitening_op = np.eye(whitening_eigvals.size, dtype='float32') # Identity
    # TODO: shouldn't it be -->  whitening_op = np.diag(1. / np.sqrt(whitening_eigvals + np.eye(whitening_eigvals.size, dtype='float32') )

if hasattr(trainset, 'data'):
    print('Selecting random patches from trainset array...')
    t = trainset.data
    n_images_trainset = t.shape[0]
    print(f'Trainset : {t.shape}')
    patches = utils.select_patches_randomly(t, patch_size=spatialsize_convolution, n_patches=n_channel_convolution,
                                            seed=args.numpy_seed, image_channels=args.input_channels)
    patches = patches.astype('float64')
    patches /= 255.0
    print(f'patches randomly selected: {patches.shape}, mean {patches.mean()} std {patches.std()}')

else:
    print('Selecting random patches from loader...')
    n_images_trainset = len(trainloader.dataset)

    if args.dataset == 'cifar10':
        trainset_select_patches = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        trainloader_select_patches = torch.utils.data.DataLoader(trainset_select_patches, batch_size=args.batchsize,
                                                                 shuffle=False, num_workers=args.num_workers)
    elif args.dataset in ['imagenet32', 'imagenet64', 'imagenet128']:
        trainset_select_patches = Imagenet32(args.path_train, transform=transforms.ToTensor(), sz=spatial_size,
                                             n_arrays=n_arrays_train)
        trainloader_select_patches = torch.utils.data.DataLoader(
            trainset_select_patches, batch_size=args.batchsize, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
    elif args.dataset in ['imagenet']:
        stride = 2
        spatial_size = 64
        traindir = os.path.join(args.path, 'train')
        trainset_select_patches = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]))
        trainloader_select_patches = torch.utils.data.DataLoader(
            trainset_select_patches, batch_size=args.batchsize, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
    elif args.dataset == 'DTD':
        trainloader_select_patches = trainloader_norandom

    n_patches_per_rowcol = spatial_size - spatialsize_convolution + 1  # 20 - 4 + 1
    patches = utils.select_patches_from_loader(loader=trainloader_select_patches,
                                               batchsize=args.batchsize,
                                               patch_size=spatialsize_convolution,
                                               n_patches=n_channel_convolution,
                                               n_images=n_images_trainset,
                                               n_patches_per_rowcol=n_patches_per_rowcol,
                                               image_channels=args.input_channels,
                                               func=None, seed=args.numpy_seed, stride=1).numpy().astype('float64')

    print(f'patches randomly selected: {patches.shape}, mean {patches.mean()} std {patches.std()}')

orig_shape = patches.shape  # (number of filters, 1, 4, 4)
patches = patches.reshape(patches.shape[0], -1)  # (number of filters, 16)
WTW_patches = (patches).dot(whitening_op).dot(whitening_op.T)  # whitened patches  --> X @ P @ Λ^(-1) @ P^(T) = X @ ∑^(-1). # TODO: Don't we need X @ ∑^(-1/2)??
kernel_convolution = torch.from_numpy(WTW_patches.astype('float32')).view(orig_shape)
print(f'kernel convolution shape: {kernel_convolution.shape}')

W_patches_norm_square = np.linalg.norm((patches).dot(whitening_op), axis=1) ** 2
bias_convolution = torch.from_numpy(0.5 * W_patches_norm_square.astype('float32')).view(1, -1, 1, 1)
print(f'bias convolution shape: {bias_convolution.shape}')

kernel_convolution = torch.from_numpy(WTW_patches.astype('float32')).view(orig_shape)

# print('Saving kernel and bias_convolutiona and exiting.')
# np.save('bias_convolution.npy', bias_convolution.numpy())
# np.save('kernel_convolution.npy', kernel_convolution.numpy())
# exit()

# print('loading kernel and bias_convolution')
# bias_convolution = torch.from_numpy(np.load('bias_convolution.npy'))
# kernel_convolution = torch.from_numpy(np.load('kernel_convolution.npy'))

if args.gaussian_patches:
    patches = np.random.normal(0, 1, size=patches.shape)
    kernel_convolution = torch.from_numpy(patches.dot(whitening_op.T).astype('float32')).view(orig_shape)
    patches_norm_square = np.linalg.norm(patches, axis=1) ** 2
    bias_convolution = torch.from_numpy(0.5 * patches_norm_square.astype('float32')).view(1, -1, 1, 1)

if args.no_cudnn:
    torch.backends.cudnn.enabled = False
else:
    cudnn.benchmark = True

params = []
if torch.cuda.is_available() and not args.learn_patches:
    kernel_convolution = kernel_convolution.half().cuda()
    bias_convolution = bias_convolution.half().cuda()
if args.learn_patches:
    kernel_convolution = nn.Parameter(kernel_convolution, requires_grad=True)
    bias_convolution = nn.Parameter(bias_convolution, requires_grad=True)
    params.append(kernel_convolution)
    params.append(bias_convolution)

if args.loss_type == "cross_entropy_loss":
    criterion = nn.CrossEntropyLoss()

if args.loss_type == "mse":
    criterion = nn.MSELoss()

k_neighbors = args.kneighbors if args.kneighbors > 0 else int(n_channel_convolution * args.kneighbors_fraction)

net = Net(kernel_convolution, bias_convolution, spatialsize_avg_pooling,
          stride_avg_pooling, finalsize_avg_pooling,
          k_neighbors=k_neighbors, sigmoid=args.sigmoid).to(device)

x = torch.rand(1, args.input_channels, spatial_size, spatial_size).to(device)
if torch.cuda.is_available() and not args.learn_patches:
    x = x.half()

out1, out2 = net(x)  # output after negative and positive convolution followed by pooling
if args.feat_square:
    out1 = torch.cat([out1, out1 ** 2], dim=1)
    out2 = torch.cat([out2, out1 ** 2], dim=1)


# Second convolution layer. This is optional and will only run if args.spatialsize_convolution_2 > 0
net_2 = None
if args.spatialsize_convolution_2 > 0:
    def func(x):
        if torch.cuda.is_available():
            x = x.half().cuda()
        return torch.cat(net(x), dim=1).float()


    n_patches_per_rowcol_2 = out1.size(2) - spatialsize_convolution + 1
    patches_2 = utils.select_patches_from_loader(trainloader_select_patches, args.batchsize,
                                                 args.spatialsize_convolution_2, args.n_channel_convolution_2,
                                                 n_images_trainset, n_patches_per_rowcol_2, func=func,
                                                 seed=args.numpy_seed, stride=1).numpy().astype('float64')
    print(f'patches 2 shape {patches_2.shape}')
    patches_mean_2, whitening_eigvecs_2, whitening_eigvals_2 = utils.compute_whitening_from_loader(
        trainloader_select_patches, patch_size=args.spatialsize_convolution_2, stride=1, func=func)
    print(
        f'Whitening 2 : mean {patches_mean_2.shape} eigvecs {whitening_eigvecs_2.shape}, eigvals max {whitening_eigvals_2.max()}, min {whitening_eigvals_2.min()} mean {whitening_eigvals_2.mean()}')

    orig_shape_2 = patches_2.shape
    patches_2 = patches_2.reshape(patches_2.shape[0], -1)
    print(f'patches 2 shape {patches_2.shape}')
    if args.whitening_reg_2 >= 0:
        inv_sqrt_eigvals_2 = np.diag(1. / np.sqrt(whitening_eigvals_2 + args.whitening_reg_2))
        whitening_op_2 = whitening_eigvecs_2.dot(inv_sqrt_eigvals_2).astype('float32')
    else:
        whitening_op_2 = np.eye(whitening_eigvals_2.size, dtype='float32')
    W_patches_2 = patches_2.dot(whitening_op_2)

    W_patches_2_norm_square = np.linalg.norm((patches_2).dot(whitening_op_2), axis=1) ** 2
    WTW_patches_2 = W_patches_2.dot(whitening_op_2.T)
    kernel_convolution_2 = torch.from_numpy(WTW_patches_2.astype('float32')).view(orig_shape_2)
    print(f'kernel convolution 2 shape: {kernel_convolution_2.shape}')

    bias_convolution_2 = torch.from_numpy(0.5 * W_patches_2_norm_square.astype('float32')).view(1, -1, 1, 1)
    print(f'bias convolution 2 shape: {bias_convolution_2.shape}')
    k_neighbors_2 = args.kneighbors_2 if args.kneighbors_2 > 0 else int(
        args.n_channel_convolution_2 * args.kneighbors_fraction_2)

    net_2 = Net(kernel_convolution_2, bias_convolution_2, spatialsize_avg_pooling=1,
                stride_avg_pooling=1, finalsize_avg_pooling=0,
                k_neighbors=k_neighbors_2, sigmoid=args.sigmoid_2).to(device)

    out1, out2 = net_2(torch.cat([out1, out2], dim=1).float())

print(f'Net output size: out1 {out1.shape[-3:]} out2 {out2.shape[-3:]}')

if args.resnet:
    resnet = utils.ResNet(2 * n_channel_convolution).to(device)
    params += list(resnet.parameters())
    classifier_blocks = [None, None, None, None, None, None]  # batch_norm1, batch_norm2, batch_norm, classifier1, classifier2, classifier
else:
    classifier_blocks = utils.create_classifier_blocks(out1, out2, args, params, n_classes)

print(f'Parameters shape {[param.shape for param in params]}')
print(f'N parameters : {sum([np.prod(list(param.shape)) for param in params]) / 1e6} millions')

del x, out1, out2

if torch.cuda.is_available() and not args.no_jit:
    print('optimizing net execution with torch.jit')
    if args.batchsize_net > 0:
        trial = torch.rand(args.batchsize_net // n_gpus, args.input_channels, spatial_size, spatial_size).to(device)
    else:
        trial = torch.rand(args.batchsize // n_gpus, args.input_channels, spatial_size, spatial_size).to(device)
    if torch.cuda.is_available() and not args.learn_patches:
        trial = trial.half()

    inputs = {'forward': (trial)}
    with torch.jit.optimized_execution(True):
        net = torch.jit.trace_module(net, inputs, check_trace=False, check_tolerance=False)
    del inputs
    del trial

if args.multigpu and n_gpus > 1:
    print(f'{n_gpus} gpus available, using Dataparralel for net')
    net = nn.DataParallel(net)

if args.normalize_net_outputs:
    mean_std_file = f'data/mean_std_{args.dataset}_seed{args.numpy_seed}_patchsize{spatialsize_convolution}_npatches{args.n_channel_convolution}_reg{args.whitening_reg}_kfraction{args.kneighbors_fraction}.npz'
    if not os.path.exists(mean_std_file):
        mean1, mean2, std1, std2 = compute_channel_mean_and_std(trainloader, net, n_channel_convolution,
                                                                kernel_convolution, bias_convolution, n_epochs=1,
                                                                seed=0)
        np.savez(mean_std_file, mean1=mean1.cpu().numpy(), mean2=mean2.cpu().numpy(), std1=std1.cpu().numpy(),
                 std2=std2.cpu().numpy())
        print(f'Net outputs mean and std computed and saved in file {mean_std_file}')
    mean_std = np.load(mean_std_file)
    mean1 = torch.from_numpy(mean_std['mean1']).to(device)
    mean2 = torch.from_numpy(mean_std['mean2']).to(device)
    std1 = torch.from_numpy(mean_std['std1']).to(device)
    std2 = torch.from_numpy(mean_std['std2']).to(device)


def train(epoch):
    net.train()
    batch_norm1, batch_norm2, batch_norm, classifier1, classifier2, classifier = classifier_blocks
    for bn in [batch_norm1, batch_norm2, batch_norm]:
        if bn is not None:
            bn.train()

    train_loss, total, correct = 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if torch.cuda.is_available() and not args.learn_patches:
            inputs = inputs.half()
        targets = targets.to(device)

        with torch.enable_grad() if args.learn_patches else torch.no_grad():
            if args.batchsize_net > 0:
                outputs = []
                for i in range(np.ceil(inputs.size(0) / args.batchsize_net).astype('int')):
                    start, end = i * args.batchsize_net, min((i + 1) * args.batchsize_net, inputs.size(0))
                    inputs_batch = inputs[start:end].to(device)
                    outputs.append(net(inputs_batch))
                outputs1 = torch.cat([out[0] for out in outputs], dim=0)
                outputs2 = torch.cat([out[1] for out in outputs], dim=0)  # concatening along the batch dimension
            else:
                inputs = inputs.to(device)
                outputs1, outputs2 = net(inputs)

            if net_2 is not None:
                outputs1, outputs2 = net_2(torch.cat([outputs1, outputs2], dim=1).float())

            if args.feat_square:
                outputs1 = torch.cat([outputs1, outputs1 ** 2], dim=1)  # concatening along the channel dimension
                outputs2 = torch.cat([outputs2, outputs1 ** 2], dim=1)

            if args.resnet:
                outputs = torch.cat([outputs1, outputs2], dim=1).float()
            else:
                outputs1, outputs2 = outputs1.float(), outputs2.float()

        optimizer.zero_grad()
        if args.resnet:
            outputs = resnet(outputs)
        else:
            if args.normalize_net_outputs:
                outputs1 = (outputs1 - mean1) / std1
                outputs2 = (outputs2 - mean2) / std2

            outputs, targets = utils.compute_classifier_outputs(outputs1, outputs2, targets, args, batch_norm1,
                                                                batch_norm2, batch_norm, classifier1, classifier2,
                                                                classifier,
                                                                train=True)

        # ipdb.set_trace()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if args.verbose:
            print("training:", epoch, batch_idx, inputs.shape, outputs.shape, targets.shape, loss)

        if torch.isnan(loss):

            return False, None
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = 100. * correct / total

    print('Train, epoch: {}; Loss: {:.2f} | Acc: {:.1f} ; kneighbors_fraction {:.3f}'.format(
        epoch, train_loss / (batch_idx + 1), train_acc, args.kneighbors_fraction))

    if args.skip_mouse_id == '-1':
        loss_log = Path(f"logs/{args.task_name}/all_mouse_data")
    else:
        loss_log = Path(f"logs/{args.task_name}/without_mouse_{args.skip_mouse_id}")
    loss_log.mkdir(parents=True, exist_ok=True)

    dict_to_save = {'epoch': epoch, 'loss': train_loss / (batch_idx + 1), 'acc': train_acc}
    with open(f'{loss_log}/train.csv', 'a', newline='') as csvfile:
        fieldnames = list(dict_to_save.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(dict_to_save)

    return True, train_acc


def test(epoch, loader=testloader, msg='Test',return_targets=False):
    global best_acc
    net.eval()
    batch_norm1, batch_norm2, batch_norm, classifier1, classifier2, classifier = classifier_blocks
    for bn in [batch_norm1, batch_norm2, batch_norm]:
        if bn is not None:
            bn.eval()

    test_loss, correct_top1, correct_top5, total = 0, 0, 0, 0
    outputs_list = []
    targets_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if torch.cuda.is_available() and not args.learn_patches:
                inputs = inputs.half()
            targets = targets.to(device)
            if args.batchsize_net > 0:
                outputs = []
                for i in range(np.ceil(inputs.size(0) / args.batchsize_net).astype('int')):
                    start, end = i * args.batchsize_net, min((i + 1) * args.batchsize_net, inputs.size(0))
                    inputs_batch = inputs[start:end].to(device)
                    outputs.append(net(inputs_batch))
                outputs1 = torch.cat([out[0] for out in outputs], dim=0)
                outputs2 = torch.cat([out[1] for out in outputs], dim=0)
            else:
                inputs = inputs.to(device)
                outputs1, outputs2 = net(inputs)

            if net_2 is not None:
                outputs1, outputs2 = net_2(torch.cat([outputs1, outputs2], dim=1).float())

            if args.feat_square:
                outputs1 = torch.cat([outputs1, outputs1 ** 2], dim=1)
                outputs2 = torch.cat([outputs2, outputs1 ** 2], dim=1)

            if args.resnet:
                outputs = torch.cat([outputs1, outputs2], dim=1).float()
                outputs = resnet(outputs)
            else:
                outputs1, outputs2 = outputs1.float(), outputs2.float()

                if args.normalize_net_outputs:
                    outputs1 = (outputs1 - mean1) / std1
                    outputs2 = (outputs2 - mean2) / std2

                outputs, targets = utils.compute_classifier_outputs(
                    outputs1, outputs2, targets, args, batch_norm1,
                    batch_norm2, batch_norm, classifier1, classifier2, classifier,
                    train=False)

            loss = criterion(outputs, targets)

            outputs_list.append(outputs)
            targets_list.append(targets)

            test_loss += loss.item()

            if args.loss_type == "mse":
                # custom made changes by me
                topk = (1,)
                cor_top1 = utils.correct_topk(outputs, targets, topk=topk)  # TODO: understand what this does
                correct_top1 += cor_top1[0]
                _, predicted = outputs.max(1)
                total += targets.size(0)

            else:
                # original code
                cor_top1, cor_top5 = utils.correct_topk(outputs, targets, topk=(1,3))  # TODO: understand what this does
                correct_top1 += cor_top1
                correct_top5 += cor_top5
                _, predicted = outputs.max(1)
                total += targets.size(0)

        test_loss /= (batch_idx + 1)
        acc1, acc5 = 100. * correct_top1 / total, 100. * correct_top5 / total

        print(
            f'{msg}, epoch: {epoch}; Loss: {test_loss:.2f} | Acc: {acc1:.1f} @1 {acc5:.1f} @5 ; kneighbors_fraction {args.kneighbors_fraction:.3f}')

        outputs = torch.cat(outputs_list, dim=0).cpu()
        targets = torch.cat(targets_list, dim=0).cpu()

        if args.skip_mouse_id == '-1':
            loss_log = Path(f"logs/{args.task_name}/all_mouse_data")
        else:
            loss_log = Path(f"logs/{args.task_name}/without_mouse_{args.skip_mouse_id}")
        loss_log.mkdir(parents=True, exist_ok=True)

        dict_to_save = {'epoch': epoch, 'loss': test_loss, 'acc': acc1}
        with open(f'{loss_log}/test.csv', 'a', newline='') as csvfile:
            fieldnames = list(dict_to_save.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(dict_to_save)

        
        if return_targets:
            return acc1, outputs, targets
        return acc1, outputs


hashname = hashlib.md5(str.encode(json.dumps(vars(args), sort_keys=True))).hexdigest()
if args.save_model:
    checkpoint_dir = f'checkpoints/{args.task_name}/{args.dataset}_{args.n_channel_convolution}patches_{args.spatialsize_convolution}x{args.spatialsize_convolution}/{args.optimizer}_{args.lr_schedule}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, f'{hashname}.pth.tar')
    print(f'Model will be saved at file {checkpoint_file}.')

    state = {'args': args}
    if os.path.exists(checkpoint_file):
        state = torch.load(checkpoint_file)

start_epoch = 0
if args.resume:
    state = torch.load(args.resume)
    start_epoch = state['epoch'] + 1
    print(f'Resuming from file {args.resume}, start epoch {start_epoch}...')
    if start_epoch not in learning_rates:
        closest_i = max([i for i in learning_rates if i <= start_epoch])
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=learning_rates[closest_i], weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=learning_rates[closest_i], momentum=args.sgd_momentum,
                                  weight_decay=args.weight_decay)
        optimizer.load_state_dict(state['optimizer'])

    for block, name in zip(classifier_blocks, ['bn1', 'bn2', 'bn', 'cl1', 'cl2', 'cl']):
        if block is not None:
            block.load_state_dict(state[name])
    acc, outputs = test(-1)




### AUC-ROC Curve ###
def compute_auc(pred_vec, truth_vec):  # for binary features
    """
    Takes two (-1,) dimensional vector with class-activation label [0,1] and returns the auc for that class
    :param pred_vec: ndarray(int) : prediction vector : e.g. [0,1,0,0,1,1,0]
    :param truth_vec: ndarray(int/float) : truth vector. (it's processed below to make it int) : e.g. [0,1,0,0,1,1,0]
    :return: float, ndarray, ndarray : the AUC, false positive rate and true positive rate
    """
    truth_vec = np.reshape(np.asarray(truth_vec), [-1])
    truth_ints = [int(i) for i in truth_vec]
    truth_ints = np.reshape(np.asarray(truth_ints), [-1])
    fp_rate, tp_rate, _ = roc_curve(truth_ints, pred_vec, pos_label=1)
    area = auc(fp_rate, tp_rate)
    return area, fp_rate, tp_rate

def plot_roc_(y_pred, y_true, title="", mode=None, epoch=None):
    """
    Take in two vectors (-1,) dimensional, with each item denoting the class that sample belongs to. 
    The class labels are converted to one-hot vectors. 
    The AUC-ROC is calculated for each class and plotted.
    :param y_pred: ndarray(int) : network prediction. e.g. [0,1,3,2,1,1,0]
    :param y_true: ndarray(int) : ground truth. e.g. [0,1,3,2,1,1,0]
    :param title: the plot title
    :return: 
    """
    # converting y_pred and y_true to one-hot vectors
    enc = OneHotEncoder()
    enc.fit(np.array([0,1,2,3]).reshape(-1,1))
    true = enc.transform(y_true.reshape(-1,1)).toarray().astype('uint8')
    pred = enc.transform(y_pred.reshape(-1, 1)).toarray().astype('uint8')
    true = list(zip(*true))
    pred = list(zip(*pred))
    # calculating auc for each class
    legend_str = []
    plt.subplot(111)

    # saving auc to a file
    dict_to_save = {}

    for i in range(len(true)):
        a, fp, tp = compute_auc(pred[i], true[i])  # "7" for diab
        a = round(a * 100, 2)
        print(f"AUC:{a}")
        plt.plot(fp, tp)  # roc_curve
        legend_str.append(f"FDG {i} | AUC:{a}%")

        dict_to_save[f"fdg_{i}"] = a


    if mode == "auc_vs_dict_size":
        # saving auv vs dict size history
        results_dir = Path(f'results/{args.task_name}/{args.spatialsize_convolution}')
        results_dir.mkdir(parents=True, exist_ok=True)

        dict_to_save["dictionary_size"] = args.n_channel_convolution

        with open(f'{results_dir}/auc_vs_dictionary_size_{args.nepochs}_{args.whitening_reg}.csv', 'a', newline='') as csvfile:
            # pickle.dump(scores_dict_to_save, f)
            fieldnames = list(dict_to_save.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(dict_to_save)


    if mode == "auc_vs_epochs":
        # saving auc vs epochs history
        results_dir = Path(f'results/{args.task_name}/{args.spatialsize_convolution}/{args.n_channel_convolution}')
        results_dir.mkdir(parents=True, exist_ok=True)

        dict_to_save["epoch"] = epoch

        with open(f'{results_dir}/auc_vs_epochs_{args.whitening_reg}.csv', 'a', newline='') as csvfile:
            # pickle.dump(scores_dict_to_save, f)
            fieldnames = list(dict_to_save.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(dict_to_save)




    plt.grid(True)
    plt.xlabel("FR Rate [%]")
    plt.ylabel("Recall [%]")
    plt.legend(legend_str, loc="lower right")
    plt.title(title)
    plt.plot([0, 1], [0, 1], color='#C0C0C0', linewidth=0.6)  # diagonal grid line
    plt.show()
    plt.savefig(f"./figures/auc_roc_{args.task_name}_{args.spatialsize_convolution}_{args.n_channel_convolution}_{args.nepochs}.png")


def get_auc_from_saved_model(mode, epoch=None):
    """
    Saves the AUC vs the required param
    mode= auc_vs_epochs: saves auc vs epochs
    mode= auc_vs_dict_size : saves auc vs dict size
    :param mode:
    :return:
    """
    acc, outputs, targets = test(-1, return_targets=True)
    outputs = outputs.argmax(1)
    plot_roc_(outputs, targets, mode=mode, epoch=epoch)



# ipdb.set_trace()

start_time = time.time()
best_test_acc, best_epoch = 0, -1

for i in range(start_epoch, args.nepochs):
    if i in learning_rates:
        print('new lr:' + str(learning_rates[i]))
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=learning_rates[i], weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=learning_rates[i], momentum=args.sgd_momentum,
                                  weight_decay=args.weight_decay)
        else:
            raise NotImplementedError('optimizer {} not implemented'.format(args.optimizer))
    no_nan_in_train_loss, train_acc = train(i)
    if not no_nan_in_train_loss:
        print(f'Epoch {i}, nan in loss, stopping training')
        break
    test_acc, outputs = test(i)

    if test_acc > best_test_acc:
        print(f'Best acc ({test_acc}).')
        best_test_acc = test_acc
        best_epoch = i

    if args.save_model or args.save_best_model and best_epoch == i:
        print(f'saving...')
        state.update({
            'optimizer': optimizer.state_dict(),
            'epoch': i,
            'acc': test_acc,
            'outputs': outputs,
        })
        for block, name in zip(classifier_blocks, ['bn1', 'bn2', 'bn', 'cl1', 'cl2', 'cl']):
            if block is not None:
                state.update({
                    name: block.state_dict()
                })
        torch.save(state, checkpoint_file)

    if args.loss_type != "mse":
        get_auc_from_saved_model(mode='auc_vs_epochs', epoch=i)

print(f'Best test acc. {best_test_acc} at epoch {best_epoch}/{i}')
hours = (time.time() - start_time) / 3600
print(f'Done in {hours:.1f} hours with {n_gpus} GPU')

if args.summary_file:
    with open(args.summary_file, "a+") as f:
        f.write(
            f'args: {args}, final_train_acc: {train_acc}, final_test_acc: {test_acc}, best_test_acc: {best_test_acc}\n')

if args.loss_type != "mse":
    get_auc_from_saved_model(mode='auc_vs_dict_size')

