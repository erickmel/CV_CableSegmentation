from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

def register_cable():
    register_coco_instances(
        "cable_train",
        {},
        "dataset/train/train.json",
        "dataset/train"
    )

    register_coco_instances(
        "cable_val",
        {},
        "dataset/validation/val.json",
        "dataset/validation"
    )