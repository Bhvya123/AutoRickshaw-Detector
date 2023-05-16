from PIL import Image
import os
import splitfolders
import shutil
import json

# Storing all the data in x,y,w,h format for each image.

directory = "./images/train"

file = open("./bbs/train.json")
bbs = json.load(file)
file.close()

# Trying to resolve corrupted dataset.
l = len(bbs)
tbn = list()
for i in range(len(bbs)):
    boxes = bbs[i]
    writer = list()
    for j in range(len(boxes)):
        x_max = max( max(boxes[j][0][0], boxes[j][1][0]), max(boxes[j][2][0],boxes[j][3][0]) )
        x_min = min( min(boxes[j][0][0], boxes[j][1][0]), min(boxes[j][2][0],boxes[j][3][0]) )
        y_max = max( max(boxes[j][0][1], boxes[j][1][1]), max(boxes[j][2][1],boxes[j][3][1]) )
        y_min = min( min(boxes[j][0][1], boxes[j][1][1]), min(boxes[j][2][1],boxes[j][3][1]) )    
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2     
        w = x_max - x_min 
        h = y_max - y_min
        writer.append([x_center,y_center,w,h])
    tbn.append(writer) 
print(tbn)

# Normalising the dataset labels.

for filename in os.listdir(directory):
    img = Image.open(directory + "/" + filename)
    i = filename.split('.')[0]
    width, height = img.size
    for j in tbn[int(i)]:
        j[0] /= width
        j[1] /= height
        j[2] /= width
        j[3] /= height
print(tbn) 


# Creating txts with same name as the train images and storing the bounding box information in them for each image
# in yolo format.

path = "./images/labels"
if(os.path.exists(path) == False):
    os.mkdir(path)
else:
    shutil.rmtree(path)
    os.mkdir(path)

for i in range(len(tbn)):
    file = open(path + "/" + str(i) + ".txt", "w")
    s = ""
    for j in tbn[i]:
        s += "0 " + str(j[0]) + " " + str(j[1]) + " " + str(j[2]) + " " + str(j[3]) + "\n"
    file.write(s)
    
# Splitting the training data into train and validation sets.
splitfolders.ratio("./images/", output="./output/", seed = 1337, ratio = (.8, .2))
