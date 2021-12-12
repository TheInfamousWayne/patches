import os


def auc_vs_dictionary_size():
    dict_size = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]
    for d in dict_size:
        os.system(f'python auc_kneighbors.py --n_channel_convolution {d} --batchsize 128 --dataset DTD --stride_avg_pooling 3 --spatialsize_avg_pooling 5 --finalsize_avg_pooling 6 --lr_schedule "{0:3e-3,50:3e-4,75:3e-5}" --nepochs 80 --optimizer SGD --bottleneck_dim 128 --padding_mode reflect --kneighbors_fraction 0.4 --convolutional_classifier 6 --whitening_reg 1e-3 --sgd_momentum 0.9 --batch_norm --save_model --save_best_model')

if __name__ == "__main__":
    auc_vs_dictionary_size()