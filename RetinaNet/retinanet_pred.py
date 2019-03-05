import numpy as np
import matplotlib.image as mpimg
import os
import sys

from k.keras_retinanet.models import load_model

import cv2
import matplotlib
import matplotlib.pyplot as plt
from k.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from k.keras_retinanet.utils.visualization import draw_box, draw_caption
from k.keras_retinanet.utils.colors import label_color

### used to fix some crash problem
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
labels_to_names = {0: 'Minion_Red', 1: 'Minion_Blue', 2: 'Turret_Blue', 3: 'Ashe', 4: 'Mouse', 5: 'Canon_Red', 6: 'Veigar', 7: 'Canon_Blue',8:'Turret_Red'}


def detect(path_img,path_model):
    
    matplotlib.use('Agg')
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

    # reduce white blank
    height, width, channels = image.shape 
    fig, ax = plt.subplots() 
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 

    plt.imshow(draw)
    plt.savefig('output/'+path_img.split('/')[-1]+'.jpg' ,dpi = 300)
    plt.show()
    

def main():
    image_path = sys.argv[1]
    model_path = 'pred.h5'
    detect(image_path,model_path)

if __name__ == '__main__':
    main()