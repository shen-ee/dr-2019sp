import os

def convert(path,cls,lab):
    out_cls = open(cls,'w')
    out_lab = open(lab,'w')
    list_cls = []
    list_pos = []
    g = os.walk(path)

    ### generate csv file for labels
    for root, _, file_list in g:
        for file_name in file_list:
            output = root + file_name.replace('.txt','.jpg')
            label = open(root + file_name, 'r')
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
    labels_path = 'BBOX/Labels/111/'
    csv_cls = 'classes.csv'
    csv_lab = 'labels.csv'
    convert(labels_path,csv_cls,csv_lab)

if __name__ == '__main__':
    main()