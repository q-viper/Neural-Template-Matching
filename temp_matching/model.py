import torch
import torch.nn as nn

from enum import Enum
from segmentation_models_pytorch import Unet
from copy import deepcopy
from typing import List


class EncodingCombination(Enum):
    ADDITION = "addition"
    CONCATENATION = "concatenation"
    MULTIPLICATION = "multiplication"


class CustomUnet(Unet):
    def __init__(
        self,
        unet_args: dict,
        encoding_combination: EncodingCombination = EncodingCombination.MULTIPLICATION,
    ):
        super().__init__(**unet_args)
        self.image_encoder = self.encoder
        self.query_encoder = deepcopy(self.encoder)
        self.encoding_combination = encoding_combination
        if self.encoding_combination == EncodingCombination.CONCATENATION:
            device = next(self.encoder.parameters()).device
            features = self.encoder(torch.rand(1, 3, 256, 256).to(device))
            combine_layers = []
            for feature in features:
                in_channel = feature.shape[1]
                # nh = (h+2p-f)/s+1
                # p=(s(nh-1)-h+f)/2
                # set s=1, p=(f-1)/2
                kernel_size = 3
                padding = (kernel_size - 1) // 2
                combine_layers.append(
                    nn.Conv2d(
                        in_channel * 2,
                        in_channel,
                        (kernel_size, kernel_size),
                        padding=padding,
                    ).to(device)
                )

            self.feature_combiner = combine_layers
            del combine_layers
            del feature

    def combine_features(
        self, image_features: List[torch.Tensor], query_features: List[torch.Tensor]
    ):
        combined_features = []
        for i, (imf, qrf) in enumerate(zip(image_features, query_features)):
            if self.encoding_combination == EncodingCombination.ADDITION:
                combined_features.append(imf + qrf)
            elif self.encoding_combination == EncodingCombination.MULTIPLICATION:
                combined_features.append(imf * qrf)
            elif self.encoding_combination == EncodingCombination.CONCATENATION:
                combined = torch.cat([imf, qrf], dim=1)
                fc = self.feature_combiner[i]
                if next(fc.parameters()).device != combined.device:
                    self.feature_combiner[i] = fc.to(combined.device)
                combined_features.append(self.feature_combiner[i](combined))

            else:
                raise ValueError(
                    f"Unknown encoding combination: {self.encoding_combination}"
                )

        return combined_features

    def get_masks(self, combined_features: List[torch.Tensor]):
        decoder_output = self.decoder(*combined_features)
        masks = self.segmentation_head(decoder_output)
        return masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image, query = x[:, 0], x[:, 1]
        self.check_input_shape(image)
        self.check_input_shape(query)
        image_features = self.image_encoder(image)
        query_features = self.query_encoder(query)
        combined_features = self.combine_features(image_features, query_features)

        return self.get_masks(combined_features)
