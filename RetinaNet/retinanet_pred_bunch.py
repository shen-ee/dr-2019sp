import retinanet_pred
import sys

def main():
    image_path = sys.argv[1]
    model_path = 'pred.h5'
    retinanet_pred.detect(image_path,model_path)

if __name__ == '__main__':
    main()