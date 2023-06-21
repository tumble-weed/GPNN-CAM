
  
#!/usr/bin/env bash
## Download COCO data for benchmarking attribution methods.
#
#mkdir -p data/datasets/coco
#
#(
#  cd data/datasets/coco
#  wget --continue http://images.cocodataset.org/zips/val2014.zip
#  wget --continue http://images.cocodataset.org/annotations/annotations_trainval2014.zip
#  unzip val2014.zip
#  unzip annotations_trainval2014.zip
#)



#!/usr/bin/env bash
# Download PASCAL VOC data for benchmarking attribution methods.

mkdir -p data/datasets/voc/VOCdevkit

(
  cd data/datasets/voc/VOCdevkit
  wget --continue http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  wget --continue http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
  tar xf VOCdevkit_08-Jun-2007.tar
  tar xf VOCtest_06-Nov-2007.tar
  mv VOCdevkit VOCdevkit_2007
  ln -s VOCdevkit_2007/VOC2007 VOC2007
)
