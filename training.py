from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="D:\yolov8\dataset.yaml", epochs=3, imgsz=640, project ="D:\yolov8", name = "train")

print(">>>>>",results)

# from ultralytics import YOLO

# model = YOLO("yolov8n.pt")  # pass any model type
# results = model.train(epochs=5)