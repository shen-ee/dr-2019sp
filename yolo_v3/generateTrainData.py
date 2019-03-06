import os
img_path  = "data/images"
label_path = "data/labels"
img_files = os.listdir(img_path)
label_files = os.listdir(label_path)
with open("train.txt", 'w') as f1:
    for i in range(len(img_files)):
        img = img_files[i]
        label = label_files[i]
        f1.write(img_path + "/" + img + " ")
        with open(label_path+"/"+label, 'r') as f2:
            f2_lines = f2.readlines()
            for j, line in enumerate(f2_lines):
                if len(line) > 5:
                    line = line.strip().replace(' ', ',')
                    line = line.replace('Mouse', '1')
                    line = line.replace('SuperMinion_Red', '3')
                    line = line.replace('Minion_Red', '2')
                    line = line.replace('Canon_Red', '4')
                    line = line.replace('Turret_Red', '5')
                    line = line.replace('Inhib_Red', '6')
                    line = line.replace('Nexus_Red', '7')
                    line = line.replace('SuperMinion_Blue', '9')
                    line = line.replace('Minion_Blue', '8')
                    line = line.replace('Canon_Blue', '10')
                    line = line.replace('Turret_Blue', '11')
                    line = line.replace('Inhib_Blue', '12')
                    line = line.replace('Nexus_Blue', '13')
                    line = line.replace('Ashe', '14')
                    line = line.replace('Veigar', '0')
                    f1.write(line)
                    if j != len(f2_lines)-1:
                        f1.write(" ")
        if i != len(img_files)-1:
            f1.write("\n")
