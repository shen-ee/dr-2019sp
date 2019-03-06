import retinanet_pred
import sys
import os

def main():
    
    path = sys.argv[1]
    g = os.walk(path)
    for root, _, file_list in g:
        for file_name in file_list:
            image_path = root + '/' + file_name
            model_path = 'pred.h5'
            retinanet_pred.detect(image_path,model_path)

if __name__ == '__main__':
    main()