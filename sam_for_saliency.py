
import sys
import numpy as np
import cv2
import dutils
import os
from benchmark.ground_truth_handler import get_gt
from matplotlib import pyplot as plt
#================================
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))    

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def visualize(image,savename,input_box=None,masks=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if masks is not None:
        show_mask(masks[0], plt.gca())
    if input_box is not None:
        show_box(input_box, plt.gca())
    plt.axis('off')
    plt.draw()
    plt.savefig(os.path.join(dutils.SAVE_DIR,savename))
    plt.close()

#================================    
def setup_sam(
    # device = "cuda",
    device = "cpu"):
    if 'sam_predictor' not in globals():
        sys.path.append("/root/evaluate-saliency-4/fong-invert/segment-anything")

        from segment_anything import sam_model_registry, SamPredictor
        sam_checkpoint = "/root/evaluate-saliency-4/fong-invert/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)    
        globals()['sam_predictor'] = predictor

def get_sam_segmentation(image,imroot,dataset,device='cpu'):
    setup_sam(
    device = device)
    global sam_predictor
    bbox_info,target_ids,classnames = get_gt(imroot,dataset)
    input_box= bbox_info['annotation']['object'][0]['bndbox']
    input_box= [int(input_box['xmin']),int(input_box['ymin']),int(input_box['xmax']),int(input_box['ymax'])]
    input_box=np.array(input_box)    
    sam_predictor.set_image(image)
    masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    visualize(image,'with_mask.png',input_box=input_box,masks=masks)
    return masks
def calculate_iou(ground_truth, predicted):
    intersection = np.logical_and(ground_truth, predicted)
    union = np.logical_or(ground_truth, predicted)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def test():
    #================================
    impath = '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000006.JPEG'
    dataset = 'imagenet'
    imroot = os.path.splitext(os.path.basename(impath.rstrip(os.path.sep)))[0]
    image = cv2.imread(impath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # input_box = np.array([100, 100, 200, 200])
    visualize(image,'with_box.png',input_box=input_box,masks=None)

    masks = get_sam_segmentation(image,imroot,dataset,device='cpu')
    print(
        calculate_iou(masks[0], masks[0])
    )
    #================================
if __name__ == '__main__':
    test()