import os


def auc_vs_dictionary_size():
    dict_size = [2500, 2000, 1500, 1000, 500, 200, 100, 80, 50, 30, 20, 10]
    patch_size = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for d in dict_size:
        for p in patch_size:
            os.system(
                'python auc_kneighbors.py \
                --n_channel_convolution %s \
                --spatialsize_convolution %s \
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
                --save_best_model' % (d, p))




def run_multiple_tasks():

    tasks = ['m_alpha_aaa', 'm_glu', 'm_h1', 'm_orn', 'm_putrescine', 'm_taurine']

    for task_name in tasks:
        try:
            os.system('python auc_kneighbors.py \
            --task_name %s \
            --loss_type "mse" \
            --n_channel_convolution 2048 \
            --batchsize 64 \
            --dataset DTD \
            --stride_avg_pooling 3 \
            --spatialsize_avg_pooling 5 \
            --finalsize_avg_pooling 6 \
            --lr_schedule "{0:3e-3,50:3e-4,75:3e-5}" \
            --nepochs 80 \
            --optimizer SGD \
            --bottleneck_dim 32 \
            --padding_mode reflect \
            --kneighbors_fraction 0.4 \
            --convolutional_classifier 6 \
            --whitening_reg 1e-1 \
            --sgd_momentum 0.9 \
            --batch_norm \
            --save_model \
            --save_best_model \
            --relu_after_bottleneck  \
            --bn_after_bottleneck \
            --verbose' % task_name)

        except Exception as e:
            print("************************ ERROR FOR %s ************************" % task_name)
            continue



def skip_mouse_results():
    tasks = ['m_alpha_aaa', 'm_glu', 'm_h1', 'm_orn', 'm_putrescine', 'm_taurine']
    mouse_id = [4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,29]

    for task_name in tasks:
        for m in mouse_id:
            try:
                os.system('python auc_kneighbors.py \
                    --task_name %s \
                    --loss_type "mse" \
                    --n_channel_convolution 2048 \
                    --batchsize 64 \
                    --skip_mouse_id %s \
                    --dataset DTD \
                    --stride_avg_pooling 3 \
                    --spatialsize_avg_pooling 5 \
                    --finalsize_avg_pooling 6 \
                    --lr_schedule "{0:3e-3,50:3e-4,75:3e-5}" \
                    --nepochs 80 \
                    --optimizer SGD \
                    --bottleneck_dim 32 \
                    --padding_mode reflect \
                    --kneighbors_fraction 0.4 \
                    --convolutional_classifier 6 \
                    --whitening_reg 1e-1 \
                    --sgd_momentum 0.9 \
                    --batch_norm \
                    --save_model \
                    --save_best_model \
                    --relu_after_bottleneck  \
                    --bn_after_bottleneck \
                    --verbose' % (task_name, str(m).zfill(3)))
            except Exception as e:
                print("************************ ERROR FOR %s %s ************************" % (task_name, str(m).zfill(3)))
                continue


if __name__ == "__main__":
    # auc_vs_dictionary_size()
    # run_multiple_tasks()
    skip_mouse_results()
    print("!!!!!DONE!!!!!")
