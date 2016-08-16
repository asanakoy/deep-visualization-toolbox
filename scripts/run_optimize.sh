layer_name='conv5'
n=256
for ((i=0; i<n;  i++)); do
    echo "--$i"
    ./optimize_image.py --push-layer $layer_name --push-channel $i --decay 0.0001 --blur-radius 1.0 --blur-every 4  --max-iter 1000 --lr-policy constant --lr-params "{'lr': 100.0}"
done
