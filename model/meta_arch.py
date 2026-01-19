import torch
import torch.nn.functional as F
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling import META_ARCH_REGISTRY
from model.line_head import LineRegressionHead

@META_ARCH_REGISTRY.register()
class CableRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        out_channels = self.backbone.output_shape()["res5"].channels
        self.line_head = LineRegressionHead(out_channels)
        self.line_loss_weight = cfg.MODEL.LINE_LOSS_WEIGHT

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        results = super().forward(batched_inputs)

        line_pred = self.line_head(features["res5"])

        if self.training:
            gt = torch.stack([x["line_params"] for x in batched_inputs]).to(
                line_pred.device
            )

            loss_line = F.smooth_l1_loss(line_pred, gt)
            results["loss_line"] = loss_line * self.line_loss_weight
        else:
            results["line_params"] = line_pred

        return results
