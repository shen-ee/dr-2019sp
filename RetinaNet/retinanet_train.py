import os
import sys
import matplotlib.image as mpimg

def convert(model_train,model_pred):
    script_convert = "python k/keras_retinanet/bin/convert_model.py "

    command_convert = " ".join([script_convert, model_train, model_pred])
    # print(command_convert)
    os.system(command_convert)


def train(labels, classes, weights, min_side, 
        max_size, epoch = 50, step = 100, lr = 1e-5):
    ### Windows
    script_train = "python k/keras_retinanet/bin/train.py --workers=0 --gpu 0 "
    
    ### Linux
    # script_train = "k/keras_retinanet/bin/train.py"
    # script_convert = "k/keras_retinanet/bin/convert_model.py"
    
    learning_rate = "--lr " + str(lr)
    steps = "--steps " + str(step)
    epochs = "--epochs " + str(epoch)
    csv_files = "csv " + labels + " " + classes
    image_min_side = "--image-min-side " + str(min_side)
    image_max_side = "--image-max-side " + str(max_size)
    command_train = " ".join([script_train, learning_rate, steps, epochs, 
                            image_min_side, image_max_side, weights, csv_files])
    # print(command_train)
    os.system(command_train)

    model_train = "snapshots/resnet50_csv_" + "%02d" % epoch + ".h5"
    model_pred = "pred.h5"
    convert(model_train,model_pred)
    

def main():

    ### set output path of converted labels
    labels_output_path = "C:/Users/sheny/Documents/GitHub/dr-2019sp/labels.csv"
    classes_output_path = "C:/Users/sheny/Documents/GitHub/dr-2019sp/classes.csv"

    width, height, num = 1920,1080,1000
     
    ### training from no-weight model, slower(need at least 10 epochs, but do not need to download pretrained model.)
    # train(step = 129, epoch = 10, lr = 1e-3,
    #     labels = labels_output_path, classes = classes_output_path,
    #     min_side = height, max_size = width, weights = "--no-weights")

    ### training from pretrained resnet50(in default), faster(do not need to train all the parameters, and 1 epoch is good enough to finish this task, but need to download pretrained model.)
    train(step = num, epoch = 50, lr = 1e-4, 
        labels = labels_output_path, classes = classes_output_path,
        min_side = height, max_size = width, weights = "--freeze-backbone")

if __name__ == '__main__':
    # model_train = "snapshots/resnet50_csv_21.h5"
    # model_pred = "pred.h5"
    # convert(model_train,model_pred)
    main()