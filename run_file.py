import os


def auc_vs_dictionary_size():
    dict_size = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]
    for d in dict_size:
        os.system(f'python kneighbors.py '
                  f'--n_channel_convolution {d} '
                  f'--batchsize 128 '
                  f'--dataset DTD '
                  f'--stride_avg_pooling 3 '
                  f'--spatialsize_avg_pooling 5 '
                  f'--finalsize_avg_pooling 6 '
                  f'--lr_schedule "{0:3e-3,50:3e-4,75:3e-5}" '
                  f'--nepochs 80 '
                  f'--optimizer SGD '
                  f'--bottleneck_dim 128 '
                  f'--padding_mode reflect '
                  f'--kneighbors_fraction 0.4 '
                  f'--convolutional_classifier 6 '
                  f'--whitening_reg 1e-3 '
                  f'--sgd_momentum 0.9 '
                  f'--batch_norm '
                  f'--save_model '
                  f'--save_best_model')
