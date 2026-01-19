from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
import torch
import math

class CableDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)

        ann = dataset_dict["annotations"][0]["polar_coordinates"][0]

        rho = ann["rho"]
        theta = ann["theta"]

        h, w = dataset_dict["image"].shape[1:]

        rho_norm = rho / math.sqrt(h*h + w*w)

        dataset_dict["line_params"] = torch.tensor([
            rho_norm,
            math.sin(theta),
            math.cos(theta)
        ], dtype=torch.float32)

        return dataset_dict
