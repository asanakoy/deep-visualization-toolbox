#!/bin/bash -x
exh_name="vanGogh_Exhibition"
cnn_dir="/net/hciserver03/storage/asanakoy/tmp/cv_init_with_mult_new.20rs_10tr_10te_128ep/${exh_name}/all"
deploy_path="$HOME/workspace/deep-visualization-toolbox/models/alexnet_exhibit/deploy.prototxt"
net_wights_path="${cnn_dir}/model/snap_iter_1240.caffemodel"

image_csv_path="/net/hciserver03/storage/asanakoy/tmp/cv_init_with_mult_new.20rs_10tr_10te_128ep/${exh_name}/all/train_list.csv"

pkl_dir="$HOME/workspace/deep-visualization-toolbox/optimize_results/${exh_name}"
max_acts_pkl_path="${pkl_dir}/max_acts.pk"

out_dir="$HOME/workspace/deep-visualization-toolbox/optimize_results/${exh_name}"

declare -a layers=("conv1" "conv2" "conv3" "conv4" "conv5" "fc6" "fc7" "fc8_exh_all")
# declare -a layers=("conv1")
for name in "${layers[@]}"
do
    extra_params=""
    if [ "${stringZ:0:4}" == "conv" ]; then
        extra_params='--do-deconv --do-deconv-norm --do-backprop --do-backprop-norm'
    fi

    ./find_maxes/crop_max_patches.py $max_acts_pkl_path $deploy_path $net_wights_path \
    --mean-path=$cnn_dir/data/mean.npy \
    --images-csv-path=$image_csv_path\
     --output-dir=$out_dir --layer=$name \
    --N=9 --gpu-id 0 \
    --do-maxes --do-info $extra_params
done
