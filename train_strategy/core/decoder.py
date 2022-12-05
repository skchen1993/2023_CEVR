import torch
import torch.nn as nn

class MLP_act(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, act=nn.ReLU) -> None:
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(act())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, act=nn.ReLU) -> None:
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        layers.append(act())
        self.layers = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class PixelShuffle_upsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
        self.conv_dimUp = nn.Conv2d(in_dim, in_dim*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_fuse = nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        # Pixel_shuffle upsample module
        x = self.conv_before_upsample(x)
        x = self.conv_dimUp(x)
        x = self.pixel_shuffle(x)
        y = self.conv_fuse(x)

        return y

class PixelShuffle_upsample_simple(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_dimUp = nn.Conv2d(in_dim, in_dim*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_fuse = nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        # Pixel_shuffle upsample module
        x = self.conv_dimUp(x)
        x = self.pixel_shuffle(x)
        y = self.conv_fuse(x)

        return y


def MLP_with_EV(feature, step, base, mlp, EV_info=1, emb=None):
    if EV_info == 2:
        feature = feature.permute(0,2,3,1)
        w, h = feature.size()[1], feature.size()[2]
        EV_list = []

        for n in range(feature.size()[0]):
            n_step = torch.full((w, h, 1), step[n].item()).float().to(feature.device)
            n_origin = torch.full((w, h, 1), base[n].item()).float().to(feature.device)
            n_EV = torch.cat([n_step, n_origin], dim=-1)
            EV_list.append(n_EV)
        EV_all = torch.stack(EV_list)
        feature = torch.cat([feature, EV_all], dim=-1)

        feature = mlp(feature)
        feature = feature.permute(0,3,1,2)
        return feature

    elif EV_info == 1:
        feature = feature.permute(0,2,3,1)
        w, h = feature.size()[1], feature.size()[2]
        EV_list = []

        for n in range(feature.size()[0]):
            n_step = torch.full((w, h, 1), step[n].item()).float().to(feature.device)
            #n_origin = torch.full((w, h, 1), base[n].item()).float().to(feature.device)
            #n_EV = torch.cat([n_step, n_origin], dim=-1)
            EV_list.append(n_step)
        EV_all = torch.stack(EV_list)
        feature = torch.cat([feature, EV_all], dim=-1)

        feature = mlp(feature)
        feature = feature.permute(0,3,1,2)
        return feature

    elif EV_info == 3:
        feature = feature.permute(0,2,3,1)
        w, h = feature.size()[1], feature.size()[2]

        EV_list = []
        for n in range(feature.size()[0]):
            n_step = torch.full((w, h, 1), step[n].item()).float().to(feature.device)
            EV_list.append(n_step)
        EV_all = torch.stack(EV_list)# (bs, h, w, 1)
        EV_emb = emb(EV_all) #(bs, h, w, 16)
        feature = torch.cat([feature, EV_emb], dim=-1) #(bs, h, w, c+16)
        feature = mlp(feature) 
        feature = feature.permute(0,3,1,2)
        return feature


class Double_conv(nn.Module):
    def __init__(self, ch, act=nn.ReLU) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act(ch),

            nn.Conv2d(ch, ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act(ch)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x

class Bottel_neck(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, mlp_num, down=None, residual=True, act=nn.ReLU, EV_info=1, emb=None) -> None:
        super().__init__()
        self.hidden_dim = 256
        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(self.hidden_dim)

        self.res = residual
        self.act = act
        self.EV_info = EV_info
        self.emb = emb


        self.downsample = down
        #self.ln = nn.LayerNorm([in_ch, h, w])
        if self.EV_info == 2:
            self.mlp = MLP(in_ch + 2, out_ch, hidden_list, act=self.act) 
        elif self.EV_info == 1:
            self.mlp = MLP(in_ch + 1, out_ch, hidden_list, act=self.act) 
        elif self.EV_info == 3:
            self.mlp = MLP(in_ch + 16, out_ch, hidden_list, act=self.act)

    def forward(self, x, s, b):
        if self.downsample != None:
            inputs = self.downsample(x)
        else:
            inputs = x

        if self.res:
            identity = inputs.clone()
            #inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s, b, self.mlp, EV_info=self.EV_info, emb=self.emb)
            return inputs + identity
        else:
            #inputs = self.ln(inputs)
            return MLP_with_EV(inputs, s, b, self.mlp, EV_info=self.EV_info, emb=self.emb)


class Upsample(nn.Module):
    def __init__(self, up_ch, in_ch, out_ch, residual=True, act=nn.ReLU) -> None:
        super().__init__()
        self.res = residual
        self.act = act
        self.upconv = nn.ConvTranspose2d(up_ch, in_ch, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( # mix up skip connection
            nn.Conv2d(in_ch *2, out_ch, kernel_size=(1, 1), stride=(1, 1)), 
            nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.act(out_ch)
        )
        self.conv3 = Double_conv(out_ch, self.act)

    def forward(self, x:torch.Tensor, y:torch.Tensor, s:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        inputs = torch.cat([x, y], dim=1)
        inputs = self.conv1(inputs)

        if self.res:
            identity = inputs.clone()
            inputs = self.conv3(inputs)
            return inputs + identity
        else:
            return self.conv3(inputs)

class Upsample_MLP(nn.Module):
    def __init__(self, up_ch, in_ch, out_ch, h, w, mlp_num=3, residual=True, act=nn.ReLU) -> None:
        super().__init__()
        self.res = residual
        self.act = act
        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(out_ch//2)

        self.upconv = nn.ConvTranspose2d(up_ch, in_ch, kernel_size=2, stride=2)
        self.ln = nn.LayerNorm([out_ch, h, w])
        self.mlp1 = MLP(in_ch *2 +2, out_ch, [], act=self.act)
        self.mlp3 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)

    def forward(self, x:torch.Tensor, y:torch.Tensor, s:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        inputs = torch.cat([x, y], dim=1)
        inputs = MLP_with_EV(inputs, s, b, self.mlp1)

        if self.res:
            identity = inputs.clone()
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s, b, self.mlp3)
            return inputs + identity
        else:
            inputs = self.ln(inputs)
            return MLP_with_EV(inputs, s, b, self.mlp3)

class Upsample_MLP_multi(nn.Module):
    def __init__(self, up_ch, in_ch, out_ch, h, w, mlp_num=3, residual=True, act=nn.ReLU) -> None:
        super().__init__()
        self.res = residual
        self.act = act
        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(out_ch//2)

        self.upconv = nn.ConvTranspose2d(up_ch, in_ch, kernel_size=2, stride=2)
        self.mlp1 = MLP(in_ch *2 +2, out_ch, [], act=self.act)
        self.ln = nn.LayerNorm([out_ch, h, w])
        self.mlp3 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
        self.mlp4 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)

    def forward(self, x:torch.Tensor, y:torch.Tensor, s:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        inputs = torch.cat([x, y], dim=1)
        inputs = MLP_with_EV(inputs, s, b, self.mlp1)

        if self.res:
            identity = inputs.clone()
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3)
            inputs = MLP_with_EV(inputs, s , b , self.mlp4)
            return inputs + identity
        else:
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3)
            return MLP_with_EV(inputs, s , b , self.mlp4)

class Upsample_MLP_multi_ResizeConvUp(nn.Module):
    def __init__(self, up_ch, in_ch, out_ch, h, w, mlp_num=3, residual=True, act=nn.ReLU, EV_info=1, emb=None) -> None:
        super().__init__()
        self.res = residual
        self.act = act
        self.EV_info = EV_info
        self.emb = emb

        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(out_ch//2)

        #self.upconv = nn.ConvTranspose2d(up_ch, in_ch, kernel_size=2, stride=2)
        # Use resize + conv3x3
        mode = 'bicubic'
        print("Resize_conv upsample mode: ", mode)
        self.resize= nn.Upsample(scale_factor=2, mode=mode)
        self.conv_resize = nn.Conv2d(up_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        if self.EV_info == 2:
            self.mlp1 = MLP(in_ch *2 +2, out_ch, [], act=self.act)
            self.ln = nn.LayerNorm([out_ch, h, w])
            self.mlp3 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
            
        elif self.EV_info == 1:
            self.mlp1 = MLP(in_ch *2 +1, out_ch, [], act=self.act)
            self.ln = nn.LayerNorm([out_ch, h, w])
            self.mlp3 = MLP(out_ch +1, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +1, out_ch, hidden_list, act=self.act)

        elif self.EV_info == 3:
            self.mlp1 = MLP(in_ch *2 +16, out_ch, [], act=self.act)
            self.ln = nn.LayerNorm([out_ch, h, w])
            self.mlp3 = MLP(out_ch +16, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +16, out_ch, hidden_list, act=self.act)        

    def forward(self, x:torch.Tensor, y:torch.Tensor, s:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
        #x = self.upconv(x)
        x = self.resize(x)
        x = self.conv_resize(x)

        inputs = torch.cat([x, y], dim=1)
        inputs = MLP_with_EV(inputs, s, b, self.mlp1, EV_info=self.EV_info, emb=self.emb)

        if self.res:
            identity = inputs.clone()
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3, EV_info=self.EV_info, emb=self.emb)
            inputs = MLP_with_EV(inputs, s , b , self.mlp4, EV_info=self.EV_info, emb=self.emb)
            return inputs + identity
        else:
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3, EV_info=self.EV_info, emb=self.emb)
            return MLP_with_EV(inputs, s , b , self.mlp4, EV_info=self.EV_info, emb=self.emb)

class Upsample_MLP_multi_PixelShuffleConvUp(nn.Module):
    def __init__(self, up_ch, in_ch, out_ch, h, w, mlp_num=3, residual=True, act=nn.ReLU, EV_info=1, emb=None) -> None:
        super().__init__()
        self.res = residual
        self.act = act
        self.EV_info = EV_info
        self.emb = emb

        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(out_ch//2)

        # Pixel_shuffle upsample module
        print("Upsample: PixelShuffle!!")
        self.PixelShuffle_upsample = PixelShuffle_upsample(up_ch, in_ch)


        if self.EV_info == 2:
            self.mlp1 = MLP(in_ch *2 +2, out_ch, [], act=self.act)
            self.ln = nn.LayerNorm([out_ch, h, w])
            self.mlp3 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
            
        elif self.EV_info == 1:
            self.mlp1 = MLP(in_ch *2 +1, out_ch, [], act=self.act)
            self.ln = nn.LayerNorm([out_ch, h, w])
            self.mlp3 = MLP(out_ch +1, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +1, out_ch, hidden_list, act=self.act)

        elif self.EV_info == 3:
            self.mlp1 = MLP(in_ch *2 +16, out_ch, [], act=self.act)
            self.ln = nn.LayerNorm([out_ch, h, w])
            self.mlp3 = MLP(out_ch +16, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +16, out_ch, hidden_list, act=self.act)        

    def forward(self, x:torch.Tensor, y:torch.Tensor, s:torch.Tensor, b:torch.Tensor) -> torch.Tensor:

        # Pixel_shuffle upsample module
        x = self.PixelShuffle_upsample(x)

        inputs = torch.cat([x, y], dim=1)
        inputs = MLP_with_EV(inputs, s, b, self.mlp1, EV_info=self.EV_info, emb=self.emb)

        if self.res:
            identity = inputs.clone()
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3, EV_info=self.EV_info, emb=self.emb)
            inputs = MLP_with_EV(inputs, s , b , self.mlp4, EV_info=self.EV_info, emb=self.emb)
            return inputs + identity
        else:
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3, EV_info=self.EV_info, emb=self.emb)
            return MLP_with_EV(inputs, s , b , self.mlp4, EV_info=self.EV_info, emb=self.emb)

class Upsample_MLP_multi_PixelShuffleConvUpSimple(nn.Module):
    def __init__(self, up_ch, in_ch, out_ch, h, w, mlp_num=3, residual=True, act=nn.ReLU, EV_info=1, emb=None) -> None:
        super().__init__()
        self.res = residual
        self.act = act
        self.EV_info = EV_info
        self.emb = emb

        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(out_ch//2)

        # Pixel_shuffle upsample module
        print("Upsample: PixelShuffle!!")
        self.PixelShuffle_upsample = PixelShuffle_upsample_simple(up_ch, in_ch)


        if self.EV_info == 2:
            self.mlp1 = MLP(in_ch *2 +2, out_ch, [], act=self.act)
            self.ln = nn.LayerNorm([out_ch, h, w])
            self.mlp3 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
            
        elif self.EV_info == 1:
            self.mlp1 = MLP(in_ch *2 +1, out_ch, [], act=self.act)
            self.ln = nn.LayerNorm([out_ch, h, w])
            self.mlp3 = MLP(out_ch +1, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +1, out_ch, hidden_list, act=self.act)

        elif self.EV_info == 3:
            self.mlp1 = MLP(in_ch *2 +16, out_ch, [], act=self.act)
            self.ln = nn.LayerNorm([out_ch, h, w])
            self.mlp3 = MLP(out_ch +16, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +16, out_ch, hidden_list, act=self.act)        

    def forward(self, x:torch.Tensor, y:torch.Tensor, s:torch.Tensor, b:torch.Tensor) -> torch.Tensor:

        # Pixel_shuffle upsample module
        x = self.PixelShuffle_upsample(x)

        inputs = torch.cat([x, y], dim=1)
        inputs = MLP_with_EV(inputs, s, b, self.mlp1, EV_info=self.EV_info, emb=self.emb)

        if self.res:
            identity = inputs.clone()
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3, EV_info=self.EV_info, emb=self.emb)
            inputs = MLP_with_EV(inputs, s , b , self.mlp4, EV_info=self.EV_info, emb=self.emb)
            return inputs + identity
        else:
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3, EV_info=self.EV_info, emb=self.emb)
            return MLP_with_EV(inputs, s , b , self.mlp4, EV_info=self.EV_info, emb=self.emb)

def build_decoder(args):
    decode_name = args.decode_name

    implemented_decoder = ('conv', 'mlp', 'mult', 'mult_resizeUp', 'mult_PixShufUp', 'mult_PixShufUpSimple')
    assert decode_name  in implemented_decoder

    decoder = None

    if decode_name == 'conv':
        decoder = Upsample

    if decode_name == 'mlp':
        decoder = Upsample_MLP

    if decode_name == 'mult':
        decoder = Upsample_MLP_multi

    if decode_name == 'mult_resizeUp':
        decoder = Upsample_MLP_multi_ResizeConvUp

    if decode_name == 'mult_PixShufUp':
        decoder = Upsample_MLP_multi_PixelShuffleConvUp

    if decode_name == 'mult_PixShufUpSimple':
        decoder = Upsample_MLP_multi_PixelShuffleConvUpSimple

    return decoder
