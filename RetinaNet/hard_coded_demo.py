import numpy as np
from k.keras_retinanet.models import load_model
from k.keras_retinanet.utils.image import resize_image, preprocess_image, read_image_bgr
import simulate_Key

import PIL.ImageGrab

import time

# set tf backend
import keras
import tensorflow as tf
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config = config)

keras.backend.tensorflow_backend.set_session(get_session())

labels_to_names = {0: 'Minion_Red', 1: 'Minion_Blue', 2: 'Turret_Blue', 3: 'Ashe', 4: 'Mouse', 5: 'Canon_Red', 6: 'Veigar', 7: 'Canon_Blue',8:'Turret_Red'}

def detect(image, model):
    # preprocess image
    image = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # compute possible bouding boxes
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time:",time.time()-start)
    box,score,label = boxes[0], scores[0], labels[0]

    # correct boxes
    boxes /= scale
    result = []
    for i in range(len(label)):
        if label[i]!=-1 and score[i] > 0.95:
            result.append([list(box[i]),score[i],label[i]])

    return result

def make_decisions(results):
    number = [0 for i in range (len(labels_to_names))]
    for item in results:
        number[item[2]] += 1
    print (number)

    direction = 1
    if number[0] >= 1 or number[8] >=1:
        direction = 0
    return direction

def simulate(direction):
    if direction == 1:
        simulate_Key.move_right()
    else:
        simulate_Key.PressKey(0x51) # q
        simulate_Key.ReleaseKey(0x51)
        simulate_Key.PressKey(0x57) # w
        simulate_Key.ReleaseKey(0x57)
        simulate_Key.move_right()    
        simulate_Key.move_left()



def main():
    time.sleep(20)
    path_model = 'pred.h5'
    model = load_model(path_model, backbone_name='resnet50')
    # while True:
    for i in range(1000):
        img = PIL.ImageGrab.grab()  
        # img.show() 
        a = time.time()
        data = detect(img, model)
        direction = make_decisions(data)
        print(time.time()-a)
        for i in data:
            print(i)
        simulate(direction)

if __name__ == '__main__':
    main()