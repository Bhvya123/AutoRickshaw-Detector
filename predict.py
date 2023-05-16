from ultralytics import YOLO
import os
import numpy as np
from PIL import Image
import shutil

model = YOLO("./runs/detect/train5/weights/best.pt") 

directory = "./test"

# Creating the directory for storing the images with bounding boxes around objects to be classified.

if(os.path.exists("./TestOutput") == False):
    os.mkdir("./TestOutput")
else:
    shutil.rmtree("./TestOutput")
    os.mkdir("./TestOutput")

textsToSubmit = list()

# Creating bounding boxes around images and storing them in TestOutput folder and generating a json file with bounding box coordinates stored.

for filename in os.listdir(directory):
    s = directory + "/" + filename
    res = model.predict(s)[0]
    ress = res.plot(line_width=5)
    ress = ress[:, :, ::-1]
    ress = Image.fromarray(ress)
    i = filename.split(".")[0]
    ress.save("./TestOutput/" + "output" + i + ".png")
    print(res.boxes.xyxy, res.boxes.conf)
    box = res.boxes.xyxy.cpu().numpy()
    box_conf = res.boxes.conf.cpu().numpy()
    appender = list()
    for i in range(len(box_conf)):
        if box_conf[i] >= 0.2:
            x1 = box[i][0]
            x2 = box[i][2]
            y1 = box[i][1]
            y2 = box[i][3]
            appender.append([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    textsToSubmit.append(appender)    

with open('test.json', 'w') as f:
    print(textsToSubmit, file=f)