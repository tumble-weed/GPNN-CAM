print('''
      compare time and MRF,LRF change when using 
      ROAD and GPNN
      ''')
import importlib
import benchmark.run_metrics
importlib.reload(benchmark.run_metrics)
from benchmark.run_metrics import main

import time
t00 = time.time()
road_metrics = main(metrics = ['road'],methodnames=['elp'],skip=False,start=0,end=1,use_images_common_to=None,delete_percentiles=
range(0,100,5),modelname='vgg16')
t10 = time.time()
print(t10-t00)
t01 = time.time()
gpnn_metrics = main(metrics = ['gpnn_eval'],methodnames=['elp'],skip=False,start=0,end=1,use_images_common_to=None,delete_percentiles=range(1,100,5),modelname='vgg16')
t11 = time.time()
print(t11-t01)