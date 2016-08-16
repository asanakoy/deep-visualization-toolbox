#!/bin/bash
exh_name="vanGogh_Exhibition"
cnn_dir="/net/hciserver03/storage/asanakoy/tmp/cv_init_with_mult_new.20rs_10tr_10te_128ep/${exh_name}/all"
deploy_path="$HOME/workspace/deep-visualization-toolbox/models/alexnet_exhibit/deploy.prototxt"
net_wights_path="${cnn_dir}/model/snap_iter_1240.caffemodel"

image_csv_path="/net/hciserver03/storage/asanakoy/tmp/cv_init_with_mult_new.20rs_10tr_10te_128ep/${exh_name}/all/train_list.csv"
out_dir="$HOME/workspace/deep-visualization-toolbox/optimize_results/${exh_name}"
out_path="${out_dir}/max_acts.pk"
mkdir -p $out_dir
./find_maxes/find_max_acts.py  $deploy_path $net_wights_path $out_path \
--images-csv-path $image_csv_path --mean-path=$cnn_dir/data/mean.npy --N 9 --gpu-id 0
