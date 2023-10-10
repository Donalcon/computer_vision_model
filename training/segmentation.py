from ultralytics import YOLO
from roboflow import Roboflow
from IPython import display
import time
import os

# set up environment
# Use an absolute path here.
abs_dataset_directory = "/home/ec2-user/possession_index/datasets"

os.environ["DATASET_DIRECTORY"] = abs_dataset_directory

rf = Roboflow(api_key="hfGn1GDEH5TgyhZuTgiG")
project = rf.workspace("donals-thesis").project("football-id-2")
dataset = project.version(6).download("yolov8")
time.sleep(5)

# Load model
model = YOLO('yolov8l-seg.pt')

# Use an absolute path for data.yaml
abs_data_yaml_path = os.path.join(abs_dataset_directory, "football-id-2-6/data.yaml")

# Train model
model.train(data=abs_data_yaml_path, epochs=50, imgsz=1080, batch=4)
