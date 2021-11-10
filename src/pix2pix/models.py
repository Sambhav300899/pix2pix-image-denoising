import torch
from torch import nn


class unet_skip_connection_block(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(
            in_channels=input_nc,
            out_channels=inner_nc,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        lrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                in_channels=inner_nc * 2,
                out_channels=outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            down = [downconv]
            up = [lrelu, upconv, nn.Tanh()]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(
                in_channels=inner_nc,
                out_channels=outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            down = [lrelu, downconv]
            up = [lrelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(
                in_channels=inner_nc * 2,
                out_channels=outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            down = [lrelu, downconv, downnorm]
            up = [lrelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class unet(nn.Module):
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_init_filters: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()

        unet_block = unet_skip_connection_block(
            outer_nc=num_init_filters * 8,
            inner_nc=num_init_filters * 8,
            input_nc=None,
            submodule=None,
            outermost=False,
            innermost=True,
            norm_layer=norm_layer,
        )

        for i in range(0, 3):
            unet_block = unet_skip_connection_block(
                outer_nc=num_init_filters * 8,
                inner_nc=num_init_filters * 8,
                input_nc=None,
                submodule=unet_block,
                outermost=False,
                innermost=False,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )

        outer_nc_mul = 4
        inner_nc_mul = 8
        for i in range(0, 3):
            unet_block = unet_skip_connection_block(
                outer_nc=num_init_filters * outer_nc_mul,
                inner_nc=num_init_filters * inner_nc_mul,
                input_nc=None,
                submodule=unet_block,
                outermost=False,
                innermost=False,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
            outer_nc_mul = outer_nc_mul // 2
            inner_nc_mul = inner_nc_mul // 2

        self.model = unet_skip_connection_block(
            outer_nc=output_nc,
            inner_nc=num_init_filters,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            innermost=False,
            norm_layer=norm_layer,
        )

    def forward(self, input):
        return self.model(input)


class patchGAN(nn.Module):
    def __init__(
        self,
        input_nc,
        num_init_filters=64,
        num_layers=3,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        layer_list = [
            nn.Conv2d(
                in_channels=input_nc,
                out_channels=num_init_filters,
                kernel_size=4,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for i in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)

            layer_list.extend(
                [
                    nn.Conv2d(
                        in_channels=num_init_filters * nf_mult_prev,
                        out_channels=num_init_filters * nf_mult,
                        kernel_size=4,
                        padding=1,
                        stride=2,
                        bias=False,
                    ),
                    nn.LeakyReLU(0.2, True),
                    norm_layer(num_init_filters * nf_mult),
                    nn.LeakyReLU(0.02, True),
                ]
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        layer_list.extend(
            [
                nn.Conv2d(
                    in_channels=num_init_filters * nf_mult_prev,
                    out_channels=num_init_filters * nf_mult,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                norm_layer(num_init_filters * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        )

        layer_list.extend(
            [
                nn.Conv2d(
                    in_channels=num_init_filters * nf_mult,
                    out_channels=1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.Sigmoid(),
            ]
        )

        self.model = nn.Sequential(*layer_list)

    def forward(self, input):
        return self.model(input)
