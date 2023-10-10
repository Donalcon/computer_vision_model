from ultralytics import YOLO
from roboflow import Roboflow
from IPython import display
display.clear_output()

rf = Roboflow(api_key="hfGn1GDEH5TgyhZuTgiG")
project = rf.workspace("donals-thesis").project("football-id-2")
dataset = project.version(5).download("yolov8")

# Load model
model = YOLO('yoloX-seg.pt')

# Train model
model.train(data='data.yaml', epochs=50, imgsz=1080)