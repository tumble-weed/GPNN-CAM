#! /bin/bash
##to_check=("gpnn-gradcam-resnet" "gpnn-gradcam-vgg16" "gpnn-gradcampp-vgg16")
to_check=("scorecam-vgg16")
date
for d in ${to_check[@]}; do 
    d=/root/bigfiles/other/results-librecam/$d
    echo $d
    ls -lh -rt $d | tail -n 5
    echo "****************************************"
done
nvidia-smi
