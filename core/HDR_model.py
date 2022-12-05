import torch
import torch.nn as nn
import pdb
from torchvision import models
from core.decoder import MLP, MLP_act, MLP_with_EV, build_decoder, Bottel_neck, Upsample_MLP
import timm


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1, bias=False):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

def brightness(img):
    img_r = img[:, 0, :, :]
    img_g = img[:, 1, :, :]
    img_b = img[:, 2, :, :]

    img_y = 0.299 * img_r + 0.587 * img_g + 0.114 * img_b
    img_y = img_y.unsqueeze(1)

    return img_y

def pad_EV(feature, step, base, EV_info=1, emb=None):
    w, h = feature.size()[2], feature.size()[3]

    EV_list = []

    if EV_info == 2:
        for n in range(feature.size()[0]):
            n_step = torch.full((1, w, h), step[n].item()).float().to(feature.device)
            n_origin = torch.full((1, w, h), base[n].item()).float().to(feature.device)
            n_EV = torch.cat([n_step, n_origin], dim=0)
            EV_list.append(n_EV)
        EV_all = torch.stack(EV_list)
        feature = torch.cat([feature, EV_all], dim=1)

        return feature

    elif EV_info == 1:
        for n in range(feature.size()[0]):
            n_step = torch.full((1, w, h), step[n].item()).float().to(feature.device)
            #n_origin = torch.full((1, w, h), base[n].item()).float().to(feature.device)
            #n_EV = torch.cat([n_step, n_origin], dim=0)
            EV_list.append(n_step)
        EV_all = torch.stack(EV_list)
        feature = torch.cat([feature, EV_all], dim=1)

        return feature       

    elif EV_info == 3:
        for n in range(feature.size()[0]):
            n_step = torch.full((1, w, h), step[n].item()).float().to(feature.device)
            EV_list.append(n_step)
        EV_all = torch.stack(EV_list)# (bs, 1, w, h)
        EV_all = EV_all.permute(0, 2, 3, 1) #(bs, w, h, 1)
        EV_emb = emb(EV_all) #(bs, w, h, 16)
        EV_emb = EV_emb.permute(0, 3, 1, 2) #(bs, 16, w, h)

        feature = torch.cat([feature, EV_emb], dim=1) #(bs, c+16, w, h)
        return feature


class MPReLU(torch.nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(MPReLU, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input):
        return -torch.nn.functional.prelu(-input, self.weight)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_parameters=' + str(self.num_parameters) + ')'
 
class HDR_UNet_BN_Res_Affine2_pad(nn.Module):
    def __init__(self, mlp_layer=2, pretrain=True, Decoder=Upsample_MLP, activation='relu', EV_info=1, init=False):
        super().__init__()
        self.act = get_activation(activation)
        self.EV_info = EV_info
        self.emb = None
        self.init = init

        if EV_info == 3:
            self.emb = MLP(1, 16, [8]) #emb DIF to dim=16 vector

        self.encoder1 = models.vgg16_bn(pretrained=pretrain).features[0:6]
        self.encoder2 = models.vgg16_bn(pretrained=pretrain).features[6:13]
        self.encoder3 = models.vgg16_bn(pretrained=pretrain).features[13:23]

        self.downsample = models.vgg16_bn(pretrained=pretrain).features[23]
        self.neck = Bottel_neck(256, 256, 32, 32, mlp_layer, self.downsample, act=self.act, EV_info=self.EV_info, emb=self.emb)

        self.decoder3 = Decoder(256, 256, 256, 64, 64, mlp_layer, act=self.act, EV_info=self.EV_info, emb=self.emb)
        self.decoder4 = Decoder(256, 128, 128, 128, 128, mlp_layer, act=self.act, EV_info=self.EV_info, emb=self.emb)
        self.decoder5 = Decoder(128, 64, 64, 256, 256, mlp_layer, act=self.act, EV_info=self.EV_info, emb=self.emb)

        if self.EV_info == 2:
            self.conv_rgb = nn.Conv2d(66, 1, kernel_size=(1, 1), stride=(1, 1))
            self.conv_mul = nn.Conv2d(66, 1, kernel_size=(1, 1), stride=(1, 1))

        elif self.EV_info == 1:
            self.conv_rgb = nn.Conv2d(65, 1, kernel_size=(1, 1), stride=(1, 1))
            self.conv_mul = nn.Conv2d(65, 1, kernel_size=(1, 1), stride=(1, 1))

        elif self.EV_info == 3:
            self.conv_rgb = nn.Conv2d(80, 1, kernel_size=(1, 1), stride=(1, 1))
            self.conv_mul = nn.Conv2d(80, 1, kernel_size=(1, 1), stride=(1, 1))


        self.encoders = [self.encoder1, self.encoder2, self.encoder3]
        self.decoders = [self.decoder3, self.decoder4, self.decoder5]

        #Initialize the weight
        if self.init == True:
            self.initialize_weights()
            print("Model weight initialization successfully!!!!")

    def forward(self, x, step, origin):
        #pdb.set_trace()

        in_img = x.clone()
        feature_maps = []
        for encoder in self.encoders:
            x = encoder(x)
            feature_maps.append(x)

        feature = self.neck(x, step, origin)

        feature_maps.reverse()
        for decoder, maps in zip(self.decoders, feature_maps):
            feature = decoder(feature, maps, step, origin)

        feature = pad_EV(feature, step, origin, EV_info=self.EV_info, emb=self.emb)


        scaler = self.conv_mul(feature)
        feature = self.conv_rgb(feature)

        return feature + in_img * scaler

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear): 
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
         
class HDR_UNet_BN_Res_Affine2_pad_Noaffine(nn.Module):
    def __init__(self, mlp_layer=2, pretrain=True, Decoder=Upsample_MLP, activation='relu', EV_info=1, init=False):
        super().__init__()
        self.act = get_activation(activation)
        self.EV_info = EV_info
        self.emb = None
        self.init = init

        if EV_info == 3:
            self.emb = MLP(1, 16, [8]) #emb DIF to dim=16 vector

        self.encoder1 = models.vgg16_bn(pretrained=pretrain).features[0:6]
        self.encoder2 = models.vgg16_bn(pretrained=pretrain).features[6:13]
        self.encoder3 = models.vgg16_bn(pretrained=pretrain).features[13:23]

        self.downsample = models.vgg16_bn(pretrained=pretrain).features[23]
        self.neck = Bottel_neck(256, 256, 32, 32, mlp_layer, self.downsample, act=self.act, EV_info=self.EV_info, emb=self.emb)

        self.decoder3 = Decoder(256, 256, 256, 64, 64, mlp_layer, act=self.act, EV_info=self.EV_info, emb=self.emb)
        self.decoder4 = Decoder(256, 128, 128, 128, 128, mlp_layer, act=self.act, EV_info=self.EV_info, emb=self.emb)
        self.decoder5 = Decoder(128, 64, 64, 256, 256, mlp_layer, act=self.act, EV_info=self.EV_info, emb=self.emb)

        """
        if self.EV_info == 2:
            self.conv_rgb = nn.Conv2d(66, 1, kernel_size=(1, 1), stride=(1, 1))
            self.conv_mul = nn.Conv2d(66, 1, kernel_size=(1, 1), stride=(1, 1))

        elif self.EV_info == 1:
            self.conv_rgb = nn.Conv2d(65, 1, kernel_size=(1, 1), stride=(1, 1))
            self.conv_mul = nn.Conv2d(65, 1, kernel_size=(1, 1), stride=(1, 1))

        elif self.EV_info == 3:
            self.conv_rgb = nn.Conv2d(80, 1, kernel_size=(1, 1), stride=(1, 1))
            self.conv_mul = nn.Conv2d(80, 1, kernel_size=(1, 1), stride=(1, 1))
        """
        self.conv_rgb = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))

        self.encoders = [self.encoder1, self.encoder2, self.encoder3]
        self.decoders = [self.decoder3, self.decoder4, self.decoder5]

        #Initialize the weight
        if self.init == True:
            self.initialize_weights()
            print("Model weight initialization successfully!!!!")

    def forward(self, x, step, origin):
        #pdb.set_trace()

        in_img = x.clone()
        feature_maps = []
        for encoder in self.encoders:
            x = encoder(x)
            feature_maps.append(x)

        feature = self.neck(x, step, origin)

        feature_maps.reverse()
        for decoder, maps in zip(self.decoders, feature_maps):
            feature = decoder(feature, maps, step, origin)

        #feature = pad_EV(feature, step, origin, EV_info=self.EV_info, emb=self.emb)
        feature = self.conv_rgb(feature)

        return feature 

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear): 
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)




def build_network(args):
    """Builds the neural network."""
    net_name = args.model_name

    implemented_networks = ('affine2_pad', 'affine2_pad_Noaff')
    assert net_name in implemented_networks

    pretraining = False
    if args.pretrain == 'vgg':
        pretraining = True
    decoder = build_decoder(args)

    net = None

    if net_name == 'affine2_pad':
        net = HDR_UNet_BN_Res_Affine2_pad(pretrain=pretraining, mlp_layer=args.mlp_num, Decoder=decoder, activation=args.act, EV_info=args.EV_info, init=args.init_weight)

    if net_name == 'affine2_pad_Noaff':
        net = HDR_UNet_BN_Res_Affine2_pad_Noaffine(pretrain=pretraining, mlp_layer=args.mlp_num, Decoder=decoder, activation=args.act, EV_info=args.EV_info, init=args.init_weight)

    print('model_name:', args.model_name, 
        'pretrain:', args.pretrain, 'mlp_num:', args.mlp_num, 
        'decoder:', args.decode_name, 'activation:', args.act)
    return net

def get_activation(name):
    implemented = ('relu', 'leaky_relu', 'gelu', 'sigmoid', 'prelu', 'mprelu')
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

    if name == 'prelu':
        act = nn.PReLU

    if name == 'mprelu':
        act = MPReLU

    return act



