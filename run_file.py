import os


def auc_vs_dictionary_size():
    dict_size = [4000, 3500, 3000, 2500, 2000, 1500, 1000, 500, 200, 100, 80, 50, 30, 20, 10]
    patch_size = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for d in dict_size:
        for p in patch_size:
            os.system(
                f'python auc_kneighbors.py \
                --n_channel_convolution {d} \
                --spatialsize_convolution {p} \
                --batchsize 128 \
                --dataset DTD \
                --stride_avg_pooling 3 \
                --spatialsize_avg_pooling 5 \
                --finalsize_avg_pooling 6 \
                --lr_schedule "{0:3e-3,50:3e-4,75:3e-5}" \
                --nepochs 80 \
                --optimizer SGD \
                --bottleneck_dim 128 \
                --padding_mode reflect \
                --kneighbors_fraction 0.4 \
                --convolutional_classifier 6 \
                --whitening_reg 1e-3 \
                --sgd_momentum 0.9 \
                --batch_norm \
                --save_model \
                --save_best_model')

if __name__ == "__main__":
    auc_vs_dictionary_size()