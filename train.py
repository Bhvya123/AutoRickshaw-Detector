from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
# Train the model
model.train(
    data='trainer.yaml', 
    epochs=100,
    batch=16,
    optimizer="SGD",
    lr0=1e-2,
    )