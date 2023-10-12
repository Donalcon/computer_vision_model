from roboflow import Roboflow

rf = Roboflow(api_key="hfGn1GDEH5TgyhZuTgiG")
project = rf.workspace("donals-thesis").project("football-id-2")
dataset = project.version(6).download("yolov8")

