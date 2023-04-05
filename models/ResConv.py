from typing import Callable
import torch
from torch import nn


class ResNet(nn.Module):
    def __init__(
        self,
        encoding_out_channels: int,
        res_in_channels: int,
        res_out_channels: int,
        n_one_hots: int,
        n_others: int,
        n_map_features: int,
        one_hot_num_classes: list,
        h: int,
        w: int,
    ) -> None:
        super(ResNet, self).__init__()
        self.encoding_input = EncodingInputLayer(
            encoding_out_channels, n_one_hots, n_others, one_hot_num_classes, h, w
        )
        self.conv1 = nn.Conv2d(
            encoding_out_channels + n_map_features, res_in_channels, kernel_size=1
        )
        self.res_block = ResidualBlock(
            res_in_channels,
            res_out_channels,
            h,
            w,
            kernel_size=5,
            activation=nn.LeakyReLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_info = x[:, :1, :, :]
        map_feature = x[:, 1:, :, :]

        global_info = self.encoding_input(global_info)
        x = torch.cat([global_info, map_feature], 1)
        x = self.conv1(x)
        x = self.res_block(x)

        return x


class EncodingInputLayer(nn.Module):
    def __init__(
        self,
        out_channels: int,
        n_one_hots: int,
        n_others: int,
        one_hot_num_classes: list,
        h: int,
        w: int,
        device="cuda",
    ):
        super(EncodingInputLayer, self).__init__()
        self.n_one_hots = n_one_hots
        self.n_others = n_others
        self.one_hot_num_classes = one_hot_num_classes
        self.h = h
        self.w = w
        one_hot_embedding = 9
        self.device = device

        self.fc = nn.Linear(sum(one_hot_num_classes), one_hot_embedding)
        self.one_hot_conv = nn.Conv2d(
            one_hot_embedding, one_hot_embedding, kernel_size=1
        )
        self.others_conv = nn.Conv2d(n_others, n_others, kernel_size=1)
        self.all_conv = nn.Conv2d(
            one_hot_embedding + n_others, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = x.reshape((b, -1))
        one_hot = x[:, : self.n_one_hots]
        others = x[:, self.n_one_hots : self.n_one_hots + self.n_others]

        one_hot_vec = torch.Tensor().to(self.device)
        for one_hot_batch in one_hot:
            one_hot_vec_batch = torch.Tensor().to(self.device)
            for j, num_classes in enumerate(self.one_hot_num_classes):
                one_hot_vec_batch = torch.cat(
                    [
                        one_hot_vec_batch,
                        nn.functional.one_hot(
                            one_hot_batch[j].to(torch.int64), num_classes
                        ),
                    ],
                    0,
                )
            one_hot_vec = torch.cat(
                [one_hot_vec, one_hot_vec_batch.reshape((1, -1))],
                0,
            )

        one_hot_vec = self.fc(one_hot_vec)

        one_hot_vec = one_hot_vec.expand((self.h, self.w) + one_hot_vec.shape).reshape(
            (b, -1, self.h, self.w)
        )
        others = others.expand((self.h, self.w) + others.shape).reshape(
            (b, -1, self.h, self.w)
        )

        one_hot_vec = self.one_hot_conv(one_hot_vec)
        others = self.others_conv(others)

        all = torch.cat([one_hot_vec, others], 1)
        all = self.all_conv(all)

        return all


class SELayer(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Average feature planes
        y = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        y = self.fc(y.view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_size: int = 3,
        normalize: bool = False,
        activation: Callable = nn.ReLU,
        squeeze_excitation: bool = True,
        **conv2d_kwargs,
    ):
        super(ResidualBlock, self).__init__()

        # Calculate "same" padding
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://www.wolframalpha.com/input/?i=i%3D%28i%2B2x-k-%28k-1%29%28d-1%29%2Fs%29+%2B+1&assumption=%22i%22+-%3E+%22Variable%22
        assert "padding" not in conv2d_kwargs.keys()
        k = kernel_size
        d = conv2d_kwargs.get("dilation", 1)
        s = conv2d_kwargs.get("stride", 1)
        padding = (k - 1) * (d + s - 1) / (2 * s)
        assert padding == int(
            padding
        ), f"padding should be an integer, was {padding:.2f}"
        padding = int(padding)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            **conv2d_kwargs,
        )
        # We use LayerNorm here since the size of the input "images" may vary based on the board size
        self.norm1 = (
            nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        )
        self.act1 = activation()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            **conv2d_kwargs,
        )
        self.norm2 = (
            nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        )
        self.final_act = activation()

        if in_channels != out_channels:
            self.change_n_channels = nn.Conv2d(in_channels, out_channels, (1, 1))
        else:
            self.change_n_channels = nn.Identity()

        if squeeze_excitation:
            self.squeeze_excitation = SELayer(out_channels)
        else:
            self.squeeze_excitation = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.act1(self.norm1(x))
        x = self.conv2(x)
        x = self.squeeze_excitation(self.norm2(x))
        x = x + self.change_n_channels(identity)
        return self.final_act(x)


class ActorOutput(nn.Module):
    def __init__(
        self, res_out_channels: int, n_robots_action: int, n_factories_action: int
    ):
        super(ActorOutput, self).__init__()
        self.spectral_norm = nn.utils.spectral_norm(
            nn.Conv2d(res_out_channels, res_out_channels, kernel_size=1)
        )
        self.robots_action = nn.Conv3d(res_out_channels, n_robots_action, kernel_size=1)
        self.robots_action_amount = nn.Conv3d(res_out_channels, 2, kernel_size=1)
        self.factories_action = nn.Conv2d(
            res_out_channels, n_factories_action, kernel_size=1
        )

    def forward(self, x):
        x = self.spectral_norm(x)
        factories_action = self.factories_action(x)
        x = x.expand((20,) + x.shape).reshape(x.shape[:2] + (-1,) + x.shape[2:])
        robots_action = self.robots_action(x)
        robots_action_amount = self.robots_action_amount(x)

        return robots_action, robots_action_amount, factories_action


class CriticOutput(nn.Module):
    def __init__(
        self,
        res_out_channels: int,
    ):
        super(CriticOutput, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size=48)
        self.critic_value = nn.Linear(res_out_channels, 1)

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.pooling(x)
        x = x.reshape((b, -1))
        return self.critic_value(x)
