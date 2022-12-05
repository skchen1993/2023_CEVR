import torch
import torch.nn as nn

from torchvision import models
from core.cycle_decoder import MLP, MLP_with_EV, build_decoder, Bottel_neck, Upsample_MLP

class HDR_UNet_BN_Res_Add(nn.Module):
    def __init__(self, mlp_layer=2, pretrain=True, Decoder=Upsample_MLP, activation='relu', rand=True, sep=0.5):
        super().__init__()
        self.act = get_activation(activation)

        self.encoder1 = models.vgg16_bn(pretrained=pretrain).features[0:6]
        self.encoder2 = models.vgg16_bn(pretrained=pretrain).features[6:13]
        self.encoder3 = models.vgg16_bn(pretrained=pretrain).features[13:23]

        self.downsample = models.vgg16_bn(pretrained=pretrain).features[23]
        self.neck = Bottel_neck(256, 256, 32, 32, mlp_layer, self.downsample, act=self.act, sep=sep)

        self.decoder3 = Decoder(256, 256, 256, 64, 64, mlp_layer, act=self.act, sep=sep)
        self.decoder4 = Decoder(256, 128, 128, 128, 128, mlp_layer, act=self.act, sep=sep)
        self.decoder5 = Decoder(128, 64, 64, 256, 256, mlp_layer, act=self.act, sep=sep)

        self.conv_rgb = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))

        self.encoders = [self.encoder1, self.encoder2, self.encoder3]
        self.decoders = [self.decoder3, self.decoder4, self.decoder5]

    def forward(self, x, step, origin):
        in_img = x.clone()
        feature_maps = []
        for encoder in self.encoders:
            x = encoder(x)
            feature_maps.append(x)

        feature = self.neck(x, step, origin)

        feature_maps.reverse()
        for decoder, maps in zip(self.decoders, feature_maps):
            feature = decoder(feature, maps, step, origin)

        feature = self.conv_rgb(feature)

        return feature + in_img

class HDR_UNet_BN_Res_Affine(nn.Module):
    def __init__(self, mlp_layer=2, pretrain=True, Decoder=Upsample_MLP, activation='relu', sep=0.5):
        super().__init__()
        self.act = get_activation(activation)

        self.encoder1 = models.vgg16_bn(pretrained=pretrain).features[0:6]
        self.encoder2 = models.vgg16_bn(pretrained=pretrain).features[6:13]
        self.encoder3 = models.vgg16_bn(pretrained=pretrain).features[13:23]

        self.downsample = models.vgg16_bn(pretrained=pretrain).features[23]
        self.neck = Bottel_neck(256, 256, 32, 32, mlp_layer, self.downsample, act=self.act, sep=sep)

        self.decoder3 = Decoder(256, 256, 256, 64, 64, mlp_layer, act=self.act, sep=sep)
        self.decoder4 = Decoder(256, 128, 128, 128, 128, mlp_layer, act=self.act, sep=sep)
        self.decoder5 = Decoder(128, 64, 64, 256, 256, mlp_layer, act=self.act, sep=sep)

        self.conv_rgb = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
        self.conv_mul = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))

        self.encoders = [self.encoder1, self.encoder2, self.encoder3]
        self.decoders = [self.decoder3, self.decoder4, self.decoder5]

    def forward(self, x, step, origin):
        in_img = x.clone()
        feature_maps = []
        for encoder in self.encoders:
            x = encoder(x)
            feature_maps.append(x)

        feature = self.neck(x, step, origin)

        feature_maps.reverse()
        for decoder, maps in zip(self.decoders, feature_maps):
            feature = decoder(feature, maps, step, origin)

        scaler = self.conv_mul(feature)
        feature = self.conv_rgb(feature)

        return feature + in_img * scaler

class HDR_UNet_BN_Res_Affine_pad(nn.Module):
    def __init__(self, mlp_layer=2, pretrain=True, Decoder=Upsample_MLP, activation='relu', sep=0.5):
        super().__init__()
        self.act = get_activation(activation)

        self.encoder1 = models.vgg16_bn(pretrained=pretrain).features[0:6]
        self.encoder2 = models.vgg16_bn(pretrained=pretrain).features[6:13]
        self.encoder3 = models.vgg16_bn(pretrained=pretrain).features[13:23]

        self.downsample = models.vgg16_bn(pretrained=pretrain).features[23]
        self.neck = Bottel_neck(256, 256, 32, 32, mlp_layer, self.downsample, act=self.act, sep=sep)

        self.decoder3 = Decoder(256, 256, 256, 64, 64, mlp_layer, act=self.act, sep=sep)
        self.decoder4 = Decoder(256, 128, 128, 128, 128, mlp_layer, act=self.act, sep=sep)
        self.decoder5 = Decoder(128, 64, 64, 256, 256, mlp_layer, act=self.act, sep=sep)

        self.conv_rgb = nn.Conv2d(66, 3, kernel_size=(1, 1), stride=(1, 1))
        self.conv_mul = nn.Conv2d(66, 3, kernel_size=(1, 1), stride=(1, 1))

        self.encoders = [self.encoder1, self.encoder2, self.encoder3]
        self.decoders = [self.decoder3, self.decoder4, self.decoder5]

    def forward(self, x, step, origin):
        in_img = x.clone()
        feature_maps = []
        for encoder in self.encoders:
            x = encoder(x)
            feature_maps.append(x)

        feature = self.neck(x, step, origin)

        feature_maps.reverse()
        for decoder, maps in zip(self.decoders, feature_maps):
            feature = decoder(feature, maps, step, origin)

        feature = pad_EV(feature, step, origin)

        scaler = self.conv_mul(feature)
        feature = self.conv_rgb(feature)

        return feature + in_img * scaler

def build_network(args):
    """Builds the neural network."""
    net_name = args.model_name

    implemented_networks = ('add', 'affine', 'affine_pad')
    assert net_name in implemented_networks

    pretraining = False
    if args.pretrain == 'vgg':
        pretraining = True
    decoder = build_decoder(args)

    net = None

    if net_name == 'add':
        net = HDR_UNet_BN_Res_Add(pretrain=pretraining, mlp_layer=args.mlp_num, Decoder=decoder, activation=args.act, sep=args.sep)

    if net_name == 'affine':
        net = HDR_UNet_BN_Res_Affine(pretrain=pretraining, mlp_layer=args.mlp_num, Decoder=decoder, activation=args.act, sep=args.sep)

    if net_name == 'affine_pad':
        net = HDR_UNet_BN_Res_Affine_pad(pretrain=pretraining, mlp_layer=args.mlp_num, Decoder=decoder, activation=args.act, sep=args.sep)

    print('model_name:', args.model_name, 
        'pretrain:', args.pretrain, 'mlp_num:', args.mlp_num, 
        'decoder:', args.decode_name, 'activation:', args.act)
    return net

def get_activation(name):
    implemented = ('relu', 'leaky_relu', 'gelu', 'sigmoid')
    assert name in implemented

    act = None

    if name == 'relu':
        act = nn.ReLU

    if name == 'leaky_relu':
        act = nn.LeakyReLU

    if name == 'gelu':
        act = nn.GELU

    if name == 'sigmoid':
        act = nn.Sigmoid

    return act

def pad_EV(feature, step, base):
    w, h = feature.size()[2], feature.size()[3]

    EV_list = []
    for n in range(feature.size()[0]):
        n_step = torch.full((1, w, h), step[n].item()).float().to(feature.device)
        n_origin = torch.full((1, w, h), base[n].item()).float().to(feature.device)
        n_EV = torch.cat([n_step, n_origin], dim=0)
        EV_list.append(n_EV)
    EV_all = torch.stack(EV_list)
    feature = torch.cat([feature, EV_all], dim=1)

    return feature

