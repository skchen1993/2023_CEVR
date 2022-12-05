import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg

from collections import namedtuple

class ContextualBilateralLoss(nn.Module):
    """
    Creates a criterion that measures the contextual bilateral loss.
    Parameters
    ---
    weight_sp : float, optional
        a balancing weight between spatial and feature loss.
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 weight_sp: float = 0.1,
                 band_width: float = 0.5,
                 loss_type: str = 'cosine',
                 use_vgg: bool = True,
                 vgg_layer: str = 'relu3_4',
                 device: int = 1,
                 patch_size : int= 8):

        super(ContextualBilateralLoss, self).__init__()

        self.band_width = band_width
        self.device = device
        self.patch_size = patch_size
        if use_vgg:
            self.vgg_model = VGG19().cuda(device = device)
            
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )
    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3,\
                'VGG model takes 3 chennel images.'

            # normalization
            x = x.sub(self.vgg_mean.detach().cuda(self.device))\
                      .div(self.vgg_std.detach().cuda(self.device))
            y = y.sub(self.vgg_mean.detach().cuda(self.device))\
                      .div(self.vgg_std.detach().cuda(self.device))

            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)

            output = self.contextual_bilateral_loss(x,y, self.band_width)
        else :
            x_patch = x.unfold(1,3,3).unfold(2, self.patch_size,self.patch_size)\
                                     .unfold(3, self.patch_size, self.patch_size)
            x_patch = x_patch.reshape([-1,self.patch_size, self.patch_size])
            x_patch = x_patch.unsqueeze(0)

            y_patch = y.unfold(1,3,3).unfold(2,self.patch_size,self.patch_size)\
                                     .unfold(3,self.patch_size,self.patch_size)
            y_patch = y_patch.reshape([-1,self.patch_size, self.patch_size])
            y_patch = y_patch.unsqueeze(0)
            
            output = self.contextual_bilateral_loss(x_patch, y_patch, 
                                                    self.band_width)
        return output

    # TODO: Operation check
    def contextual_bilateral_loss(self,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  weight_sp: float = 0.1,
                                  band_width: float = 0.5,
                                  loss_type: str = 'cosine'):
        """
        Computes Contextual Bilateral (CoBi) Loss between x and y,
            proposed in https://arxiv.org/pdf/1905.05169.pdf.
        Parameters
        ---
        x : torch.Tensor
            features of shape (N, C, H, W).
        y : torch.Tensor
            features of shape (N, C, H, W).
        band_width : float, optional
            a band-width parameter used to convert distance to similarity.
            in the paper, this is described as :math:`h`.
        loss_type : str, optional
            a loss type to measure the distance between features.
            Note: `l1` and `l2` frequently raises OOM.
        Returns
        ---
        cx_loss : torch.Tensor
            contextual loss between x and y (Eq (1) in the paper).
        k_arg_max_NC : torch.Tensor
            indices to maximize similarity over channels.
        """

        assert x.size() == y.size(), 'input tensor must have the same size.'
        # spatial loss
        grid = self.compute_meshgrid(x.shape).to(self.device)
        #print("grid.shape: ", grid.shape)
        dist_raw = self.compute_l2_distance(grid, grid)
        dist_tilde = self.compute_relative_distance(dist_raw)
        cx_sp = self.compute_cx(dist_tilde, band_width)

        # feature loss
        if loss_type == 'cosine':
            dist_raw = self.compute_cosine_distance(x, y)
        elif loss_type == 'l1':
            dist_raw = self.compute_l1_distance(x, y)
        elif loss_type == 'l2':
            dist_raw = self.compute_l2_distance(x, y)
        dist_tilde = self.compute_relative_distance(dist_raw)
        cx_feat = self.compute_cx(dist_tilde, band_width)

        # combined loss
        cx_combine = (1. - weight_sp) * cx_feat + weight_sp * cx_sp

        k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)

        cx = k_max_NC.mean(dim=1)
        cx_loss = torch.mean(-torch.log(cx + 1e-5))

        return cx_loss


    def compute_cx(self, dist_tilde, band_width):
        w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
        cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
        return cx


    def compute_relative_distance(self, dist_raw):
        dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
        dist_tilde = dist_raw / (dist_min + 1e-5)
        return dist_tilde


    def compute_cosine_distance(self, x, y):
        # mean shifting by channel-wise mean of `y`.
        y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
        x_centered = x - y_mu
        y_centered = y - y_mu

        # L2 normalization
        x_normalized = F.normalize(x_centered, p=2, dim=1)
        y_normalized = F.normalize(y_centered, p=2, dim=1)

        # channel-wise vectorization
        N, C, *_ = x.size()
        x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
        y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

        # consine similarity
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                               y_normalized)  # (N, H*W, H*W)

        # convert to distance
        dist = 1 - cosine_sim

        return dist


    # TODO: Considering avoiding OOM.
    def compute_l1_distance(self, x: torch.Tensor, y: torch.Tensor):
        N, C, H, W = x.size()
        x_vec = x.view(N, C, -1)
        y_vec = y.view(N, C, -1)

        dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
        dist = dist.sum(dim=1).abs()
        dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
        dist = dist.clamp(min=0.)

        return dist


    # TODO: Considering avoiding OOM.
    def compute_l2_distance(self, x, y):
        N, C, H, W = x.size()
        x_vec = x.view(N, C, -1)
        y_vec = y.view(N, C, -1)
        x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
        y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)

        A = y_vec.transpose(1, 2) @ x_vec
        # print(x.shape, y_s.shape, A.shape, x_s.shape)
        dist = y_s - 2 * A + x_s.transpose(1, 2)
        dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
        dist = dist.clamp(min=0.)

        return dist


    def compute_meshgrid(self, shape):
        N, C, H, W = shape
        rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
        cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

        feature_grid = torch.meshgrid(rows, cols)
        feature_grid = torch.stack(feature_grid).unsqueeze(0)
        feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

        return feature_grid

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2','relu3_4', 'relu4_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4)

        return out


"""
if __name__ == '__main__':
    bs = 64
    pred = torch.rand(bs, 3, 64, 64).to('cuda')
    gt = torch.rand(bs, 3, 64, 64).to('cuda')

    cx = ContextualBilateralLoss(use_vgg = True, vgg_layer = 'relu3_4', 
                           device = 'cuda')

    cx_loss = cx(pred, gt)

    print("cx_loss: ", cx_loss)
"""