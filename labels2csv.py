import os

def convert(inpath_labels, inpath_imgs, outpath_cls, outpath_labels):
    out_cls = open(outpath_cls,'w')
    out_lab = open(outpath_labels,'w')
    list_cls = []
    g = os.walk(inpath_labels)

    ### generate csv file for labels
    for _, _, file_list in g:
        for file_name in file_list:
            output = inpath_imgs + file_name.replace('.txt','.jpg')
            label = open(inpath_labels + file_name, 'r')
            num = int(label.readline())
            for i in range(num):
                line = label.readline()
                cls_name = line.split()[4]
                if cls_name not in list_cls:
                    list_cls.append(cls_name)
                output += ',' + line.replace('\n','').replace(' ',',')
            out_lab.write(output+'\n')
    

    ### generate csv file for classes                
    for id, name in enumerate(list_cls):
        out_cls.write(name + ',' + str(id) + '\n')

def main():
    path_labels = 'BBOX/Labels/111/'
    path_images = 'BBOX/Images/111/'
    csv_classes = 'classes.csv'
    csv_labels = 'labels.csv'
    convert(path_labels,path_images,csv_classes,csv_labels)

if __name__ == '__main__':
    main()