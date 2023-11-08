from roboflow import Roboflow
import os

abs_dataset_directory = "/Users/donalconlon/Documents/GitHub/possession_index/homography/calibration/datasets"

rf = Roboflow(api_key="hfGn1GDEH5TgyhZuTgiG")
project = rf.workspace("donals-thesis").project("football-id-2")
dataset = project.version(7).download("coco-segmentation")

