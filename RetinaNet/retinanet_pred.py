import numpy as np
import matplotlib.image as mpimg
import os
import sys

from k.keras_retinanet.models import load_model

import cv2
import matplotlib.pyplot as plt
from k.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from k.keras_retinanet.utils.visualization import draw_box, draw_caption
from k.keras_retinanet.utils.colors import label_color

### used to fix some crash problem
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}

def detect(path_img,path_model):
    ### load prediction model
    model = load_model(path_model, backbone_name='resnet50')

    ### load image

    # load image
    image = read_image_bgr(path_img)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    ### compute possible bouding boxes
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))


    for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
        if score < 0.5:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()

def main():
    image_path = sys.argv[1]
    model_path = 'pred.h5'
    detect(image_path,model_path)

if __name__ == '__main__':
    main()