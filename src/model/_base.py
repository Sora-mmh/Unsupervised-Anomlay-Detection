from abc import abstractmethod, ABC
import logging
from typing import List, Union
import sys

import torch
import torch.nn as nn

sys.path.insert(
    0,
    "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/other/Unsupervised-Anomlay-Detection/src",
)

from encoder.swin_transformer import (
    SwinTEncoder,
    SwinSEncoder,
    SwinBEncoder,
    SwinLEncoder,
)
from decoder import UnetPlusPlusDecoder


logging.basicConfig(level=logging.INFO)


class SwunetPlusPlus(nn.Module):

    # Members
    _encoder: Union[SwinTEncoder, SwinSEncoder, SwinBEncoder, SwinLEncoder]
    _decoder: UnetPlusPlusDecoder
    _encoder_channels: List[int]

    def __init__(self, encoder_channels: List[int] = [96, 192, 384, 768]):
        super().__init__()
        self._encoder_channels = encoder_channels
        self._encoder = SwinTEncoder()
        self._decoder = UnetPlusPlusDecoder(encoder_channels=self._encoder_channels)

    @property
    def encoder(self) -> Union[SwinTEncoder, SwinSEncoder, SwinBEncoder, SwinLEncoder]:
        return self._encoder

    @property
    def decoder(self) -> UnetPlusPlusDecoder:
        return self._decoder

    @abstractmethod
    def train(self) -> None:
        return NotImplemented

    @abstractmethod
    def _show_metrics(self) -> None:
        return NotImplemented


if __name__ == "__main__":
    model = SwunetPlusPlus()
    model = model.cuda()
    img = torch.rand(1, 3, 448, 448).cuda()
    features = model.encoder(img)
    features = [feature.permute(0, 3, 1, 2) for feature in features]
    output = model.decoder(features)
    print(output.size())
