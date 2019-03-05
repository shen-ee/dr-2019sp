import os
import sys

def convert(model_train,model_pred):
    script_convert = "python k/keras_retinanet/bin/convert_model.py "

    command_convert = " ".join([script_convert, model_train, model_pred])
    os.system(command_convert)

def main():
    model_train = "snapshots/resnet50_csv_08.h5"
    model_pred = "pred.h5"
    convert(model_train,model_pred)

if __name__ == '__main__':
    main()