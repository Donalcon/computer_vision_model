import supervision as sv

ANNOTATIONS_DIRECTORY_PATH_1 = "C:/Users/Owner/Downloads/dataset-20230630T102538Z-001/dataset/train/labels"
IMAGES_DIRECTORY_PATH_1 = "C:/Users/Owner/Downloads/dataset-20230630T102538Z-001/dataset/train/images"
DATA_YAML_PATH_1 = "C:/Users/Owner/Downloads/dataset-20230630T102538Z-001/dataset/data.yaml"

dataset_1 = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH_1,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH_1,
    data_yaml_path=DATA_YAML_PATH_1)

len(dataset_1)

ANNOTATIONS_DIRECTORY_PATH_2 = "C:/Users/Owner/Downloads/dataset_2-20230629T195327Z-001/dataset_2/train/labels"
IMAGES_DIRECTORY_PATH_2 = "C:/Users/Owner/Downloads/dataset_2-20230629T195327Z-001/dataset_2//train/images"
DATA_YAML_PATH_2 = "C:/Users/Owner/Downloads/dataset_2-20230629T195327Z-001/dataset_2/data.yaml"

dataset_2 = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH_2,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH_2,
    data_yaml_path=DATA_YAML_PATH_2)

len(dataset_2)

ANNOTATIONS_DIRECTORY_PATH_3 = "C:/Users/Owner/Downloads/dataset_4-20230630T105737Z-001/dataset_4/train/labels"
IMAGES_DIRECTORY_PATH_3 = "C:/Users/Owner/Downloads/dataset_4-20230630T105737Z-001/dataset_4//train/images"
DATA_YAML_PATH_3 = "C:/Users/Owner/Downloads/dataset_4-20230630T105737Z-001/dataset_4/data.yaml"

dataset_3 = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH_3,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH_3,
    data_yaml_path=DATA_YAML_PATH_3)

len(dataset_3)

merged_dataset = sv.DetectionDataset.merge([dataset_1, dataset_2, dataset_3])
