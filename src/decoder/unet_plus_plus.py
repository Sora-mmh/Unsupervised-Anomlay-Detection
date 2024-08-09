import torch
import torch.nn as nn
import torch.nn.functional as F

import modules as md

from swin_transformer_encoder import SwinTEncoder


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        skip_channels,
        output_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            input_channels + skip_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=input_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=output_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.input_channels = encoder_channels[::-1]
        self.output_channels = self.input_channels[1:]
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = {}
        self.layers = len(encoder_channels) - 1
        for layer in range(self.layers - 1):
            for depth in range(self.layers - layer):
                input_channels = self.input_channels[depth]
                skip_channels = self.input_channels[depth + 1]
                output_channels = self.output_channels[depth]
                blocks[f"Block_{depth}_{layer + 1}"] = DecoderBlock(
                    input_channels, skip_channels, output_channels, **kwargs
                )
            self.input_channels = self.input_channels[1:]
            self.output_channels = self.output_channels[1:]
        blocks[f"Block_{0}_{self.layers}"] = DecoderBlock(
            input_channels=576, skip_channels=96, output_channels=96, **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)

    def forward(self, *features):
        features = features[0]
        features = features[::-1]
        decoded_outputs = {}
        for layer in range(self.layers):
            for depth in range(self.layers - layer):
                if layer == 0:
                    decoded_outputs[f"X_{depth}_{layer + 1}"] = self.blocks[
                        f"Block_{depth}_{layer + 1}"
                    ](features[depth], features[depth + 1])
                else:
                    previous_same_depth_concatenated_outputs = [
                        decoded_outputs[f"X_{depth}_{layer}"]
                        for layer in range(1, layer + 1)
                    ]
                    previous_depth_plus_one_output = decoded_outputs[
                        f"X_{depth + 1}_{layer}"
                    ]
                    last_output_dim = previous_same_depth_concatenated_outputs[
                        -1
                    ].shape[-1]
                    for idx, output in enumerate(
                        previous_same_depth_concatenated_outputs[:-1]
                    ):
                        previous_same_depth_concatenated_outputs[idx] = F.interpolate(
                            output,
                            scale_factor=last_output_dim // output.shape[-1],
                            mode="nearest",
                        )
                    decoded_outputs[f"X_{depth}_{layer + 1}"] = self.blocks[
                        f"Block_{depth}_{layer + 1}"
                    ](
                        torch.cat(previous_same_depth_concatenated_outputs, dim=1),
                        previous_depth_plus_one_output,
                    )
        return decoded_outputs[f"X_{0}_{self.layers}"]


if __name__ == "__main__":
    encoder = SwinTEncoder()
    encoder = encoder.cuda()
    decoder = UnetPlusPlusDecoder(encoder_channels=[96, 192, 384, 768])
    decoder = decoder.cuda()
    img = torch.rand(1, 3, 448, 448).cuda()
    features = encoder(img)
    features = [feature.permute(0, 3, 1, 2) for feature in features]
    output = decoder(features)
    print(output.size())
