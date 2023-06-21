import os

dataset_dir = "/root/bigfiles/dataset/stanford-dogs-dataset"
train_dir = os.path.join(dataset_dir, "Images")

class_paths = {}
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    if os.path.isdir(class_dir):
        image_paths = [os.path.join(class_dir, filename) for filename in os.listdir(class_dir)]
        class_paths[class_name] = image_paths
        
pass