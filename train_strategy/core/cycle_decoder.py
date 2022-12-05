import torch
import torch.nn as nn
import random

class MLP(nn.Module):
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

def MLP_with_EV(feature, step, base, mlp):
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
    def __init__(self, in_ch, out_ch, h, w, mlp_num, down, sep=0.5, residual=True, act=nn.ReLU) -> None:
        super().__init__()
        self.hidden_dim = 256
        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(self.hidden_dim)

        self.res = residual
        self.act = act
        self.divide = sep

        self.downsample = down
        self.ln = nn.LayerNorm([in_ch, h, w])
        self.mlp = MLP(in_ch + 2, out_ch, hidden_list, act=self.act) # 2 for two mlp inputs

    def set_sep(self, rand, sep):
        if rand:
            self.sep = random.uniform(0., 1.)
        else:
            self.sep = sep
        self.remain = 1 - sep

    def forward(self, x, s, b):
        inputs = self.downsample(x)
        self.set_sep(self.training, self.divide)
        
        if self.res:
            identity = inputs.clone()
            #inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s * self.sep, b, self.mlp)
            inputs = MLP_with_EV(inputs, s * self.remain, b + s * self.sep, self.mlp)
            return inputs + identity
        else:
            #inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s * self.sep, b, self.mlp)
            inputs = MLP_with_EV(inputs, s * self.remain, b + s * self.sep, self.mlp)
            return inputs

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
    def __init__(self, up_ch, in_ch, out_ch, h, w, mlp_num=3, sep=0.5, residual=True, act=nn.ReLU) -> None:
        super().__init__()
        self.res = residual
        self.act = act
        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(out_ch//2)
        self.divide = sep

        self.upconv = nn.ConvTranspose2d(up_ch, in_ch, kernel_size=2, stride=2)
        self.ln = nn.LayerNorm([out_ch, h, w])
        self.mlp1 = MLP(in_ch *2 +2, out_ch, [], act=self.act)
        self.mlp3 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)

    def set_sep(self, rand, sep):
        if rand:
            self.sep = random.uniform(0., 1.)
        else:
            self.sep = sep
        self.remain = 1 - sep

    def forward(self, x:torch.Tensor, y:torch.Tensor, s:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        inputs = torch.cat([x, y], dim=1)
        inputs = MLP_with_EV(inputs, s, b, self.mlp1)
        self.set_sep(self.training, self.divide)

        if self.res:
            identity = inputs.clone()
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s * self.sep, b, self.mlp3)
            inputs = MLP_with_EV(inputs, s * self.remain, b + s * self.sep, self.mlp3)
            return inputs + identity
        else:
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s * self.sep, b, self.mlp3)
            inputs = MLP_with_EV(inputs, s * self.remain, b + s * self.sep, self.mlp3)
            return inputs

class Upsample_MLP_multi(nn.Module):
    def __init__(self, up_ch, in_ch, out_ch, h, w, mlp_num=3, sep=0.5, residual=True, act=nn.ReLU) -> None:
        super().__init__()
        self.res = residual
        self.act = act
        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(out_ch//2)
        self.divide = sep

        self.upconv = nn.ConvTranspose2d(up_ch, in_ch, kernel_size=2, stride=2)
        self.mlp1 = MLP(in_ch *2 +2, out_ch, [], act=self.act)
        self.ln = nn.LayerNorm([out_ch, h, w])
        self.mlp3 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
        self.mlp4 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)

    def set_sep(self, rand, sep):
        if rand:
            self.sep = random.uniform(0., 1.)
        else:
            self.sep = sep
        self.remain = 1 - sep

    def forward(self, x:torch.Tensor, y:torch.Tensor, s:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        inputs = torch.cat([x, y], dim=1)
        inputs = MLP_with_EV(inputs, s, b, self.mlp1)
        self.set_sep(self.training, self.divide)

        if self.res:
            identity = inputs.clone()
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s * self.sep, b, self.mlp3)
            inputs = MLP_with_EV(inputs, s * self.sep, b, self.mlp4)
            inputs = MLP_with_EV(inputs, s * self.remain, b + s * self.sep, self.mlp3)
            inputs = MLP_with_EV(inputs, s * self.remain, b + s * self.sep, self.mlp4)
            return inputs + identity
        else:
            inputs = self.ln(inputs)
            inputs = MLP_with_EV(inputs, s * self.sep, b, self.mlp3)
            inputs = MLP_with_EV(inputs, s * self.sep, b, self.mlp4)
            inputs = MLP_with_EV(inputs, s * self.remain, b + s * self.sep, self.mlp3)
            inputs = MLP_with_EV(inputs, s * self.remain, b + s * self.sep, self.mlp4)
            return inputs

def build_decoder(args):
    decode_name = args.decode_name

    implemented_decoder = ('conv', 'mlp', 'mult')
    assert decode_name  in implemented_decoder

    decoder = None

    if decode_name == 'conv':
        decoder = Upsample

    if decode_name == 'mlp':
        decoder = Upsample_MLP

    if decode_name == 'mult':
        decoder = Upsample_MLP_multi

    return decoder
