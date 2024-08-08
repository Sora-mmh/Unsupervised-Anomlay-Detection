import torch
import torch.nn as nn
import torch.nn.functional as F

import modules as md


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
        center=False,
    ):
        super().__init__()

        decoder_channels = encoder_channels[::-1]
        self.input_channels = encoder_channels
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.output_channels = decoder_channels
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = {}
        self.layers = len(encoder_channels) - 1
        for layer in range(self.layers):
            for depth in range(self.layers - layer):
                input_channel = self.input_channels[depth]
                skip_channel = self.input_channels[depth + 1]
                output_channel = input_channel
                blocks[f"Block_{depth}_{layer + 1}"] = DecoderBlock(
                    input_channel, skip_channel, output_channel, **kwargs
                )
        self.blocks = nn.ModuleDict(blocks)

    def forward(self, *features):
        features = features
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
                        decoded_outputs[f"X_{depth}_{layer + 1}"]
                        for layer in range(1, layer + 1)
                    ]
                    previous_depth_plus_one_output = decoded_outputs[
                        f"X_{depth + 1}_{layer}"
                    ]
                    decoded_outputs[f"X_{depth}_{layer + 1}"] = self.blocks[
                        f"Block_{depth}_{layer + 1}"
                    ](
                        torch.cat(previous_same_depth_concatenated_outputs),
                        previous_depth_plus_one_output,
                    )
        return decoded_outputs[f"X_{0}_{self.layers}"]


if __name__ == "__main__":
    encoder_channels = [64, 128, 256, 512]
    decoder = UnetPlusPlusDecoder(encoder_channels=encoder_channels)
    decoder = decoder.cuda()
    img = torch.rand(1, 64, 448, 448).cuda()
    output = decoder(img)
    print(output.size())
