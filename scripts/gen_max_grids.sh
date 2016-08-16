#!/bin/bash -x
exh_name="vanGogh_Exhibition"

crops_dir="$HOME/workspace/deep-visualization-toolbox/optimize_results/${exh_name}/"
out_dir="$HOME/workspace/deep-visualization-toolbox/models/alexnet_exhibit/${exh_name}/unit_jpg_vis"

declare -a layers=("conv1" "conv2" "conv3" "conv4" "conv5" "fc6" "fc7" "fc8_exh_all")
# declare -a layers=("conv1")
for name in "${layers[@]}"
do
    do_deconv=""
    if [ "${stringZ:0:4}" == "conv" ]; then
        do_deconv='--do-deconv'
    fi
    python find_maxes/gen_max_grids.py $crops_dir $out_dir --layer=$name \
    --N=9 --do-maxim $do_deconv
done
