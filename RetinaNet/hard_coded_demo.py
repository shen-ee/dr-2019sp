import numpy as np
from k.keras_retinanet.models import load_model
import simulate_Key

import PIL.ImageGrab

import time

labels_to_names = {0: 'Minion_Red', 1: 'Minion_Blue', 2: 'Turret_Blue', 3: 'Ashe', 4: 'Mouse', 5: 'Canon_Red', 6: 'Veigar', 7: 'Canon_Blue',8:'Turret_Red'}

def detect(image, model):
    ### compute possible bouding boxes
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    box,score,label = boxes[0], scores[0], labels[0]
    result = []
    for i in range(len(label)):
        if label[i]!=-1 and score[i] > 0.6:
            result.append([box[i],score[i],label[i]])

    return result

def make_decisions(results):
    direction = 1
    for item in results:
        if item[2] == 0:
            direction = 0
    return direction

def simulate(direction):
    if direction == 1:
        simulate_Key.move_right()
    else:
        simulate_Key.move_left()



def main():
    # time.sleep(20)
    path_model = 'pred.h5'
    model = load_model(path_model, backbone_name='resnet50')
    while True:
        img = PIL.ImageGrab.grab()  
        # img.show() 
        a = time.time()
        data = detect(img, model)
        direction = make_decisions(data)
        print(time.time()-a)
        print(data)
        simulate(direction)

if __name__ == '__main__':
    main()