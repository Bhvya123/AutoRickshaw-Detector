# ML Model for Auto Rickshaw Detection in Images

- Used YOLOv8 Model for Object Detection from ultralytics.
- YOLOv8 is state of the art model for real-time object detection, segmentation and classification.

## train.py:
    - Used to train the model on the datasets.

## predict.py:
    - To predict the bounding box around Auto Rickshaws on given testset images.
    - Bounding box coordinates are stored in test.json file according to specified confidence level.

## convertToYolo.py:
    - To convert the bbs/train.json to yolo format.
    - Adds normalised labels in yolo format for the training set into a labels folder in images folder.
    - And divide the dataset (images folder) into train and validation sets.

## trainer.yaml:
    - Contains dataset path.
    - Contains object class names (as per the number they are mapped to in dataset) to detect in images.

### Dataset Directory Structure:
    -dataset
        - images
            - train set
            - validation set
            - test set (optional)
        - labels
            - train set
            - validation set
            - test set (optional)

    To create dataset first run convertToYolo.py and then from the output folder add to the dataset as per directory structure shown above.
    Also update the path in trainer.yaml.

-For the ML Challenge by DataFoundation IIIT-H.