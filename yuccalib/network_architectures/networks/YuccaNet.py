import numpy as np
import torch
from torch import nn
from yuccalib.network_architectures.utils import get_steps_for_sliding_window


class YuccaNet(nn.Module):
    def __init__(self):
        super(YuccaNet, self).__init__()
        # Attributes that need to be defined for each architecture
        self.abbreviation: str = None
        self.dimensions: int = None

    def forward(self):
        """
        implement in individual trainers.
        DO NOT INCLUDE FINAL SOFTMAX/SIGMOID ETC.
        WILL BE HANDLED BY LOSS FUNCTIONS
        """
        pass

    def predict(self, mode, data, patch_size, overlap, mirror=False):
        if torch.cuda.is_available():
            data = data.to("cuda")

        self.eval()
        with torch.no_grad():
            if mode == '3D':
                #pred = self._predict3D(data, patch_size, overlap)
                predict = self._predict3D
            if mode == '2.5D':
                pred = self._predict25D(data, patch_size, overlap)
            if mode == '2D':
                pred = self._predict2D(data, patch_size, overlap)
            pred = predict(data, patch_size, overlap)
            if mirror:
                pred += torch.flip(predict(torch.flip(data, (2, )), patch_size, overlap), (2, ))
                pred += torch.flip(predict(torch.flip(data, (3,)), patch_size, overlap), (3, ))
                pred += torch.flip(predict(torch.flip(data, (2, 3)), patch_size, overlap), (2, 3))
                div = 4
                if mode == '3D':
                    pred += torch.flip(predict(torch.flip(data, (4, )), patch_size, overlap), (4, ))
                    pred += torch.flip(predict(torch.flip(data, (2, 4)), patch_size, overlap), (2, 4))
                    pred += torch.flip(predict(torch.flip(data, (3, 4)), patch_size, overlap), (3, 4))
                    pred += torch.flip(predict(torch.flip(data, (2, 3, 4)), patch_size, overlap), (2, 3, 4))
                    div += 4
                pred /= div
        return pred

    def _predict3D(self, data, patch_size, overlap):
        """
        Sliding window prediction implementation
        """
        canvas = torch.zeros((1, self.num_classes, *data.shape[2:]),
                             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        x_steps, y_steps, z_steps = get_steps_for_sliding_window(data.shape[2:], patch_size, overlap)
        px, py, pz = patch_size

        for xs in x_steps:
            for ys in y_steps:
                for zs in z_steps:
                    # check if out of bounds
                    out = self.forward(data[:, :, xs:xs+px, ys:ys+py, zs:zs+pz])
                    canvas[:, :, xs:xs+px, ys:ys+py, zs:zs+pz] += out
        return canvas

    def _predict25D(self, data, patch_size, overlap):
        pass

    def _predict2D(self, data, patch_size, overlap):
        """
        Sliding window prediction implementation
        """
        canvas = torch.zeros((1, self.num_classes, *data.shape[2:]),
                             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        px, py = patch_size

        # If we have 5 dimensions we are working with 3D data, and need to predict each slice.
        if len(data.shape) == 5:
            x_steps, y_steps = get_steps_for_sliding_window(data.shape[3:], patch_size, overlap)
            for idx in range(data.shape[2]):
                for xs in x_steps:
                    for ys in y_steps:
                        out = self.forward(data[:,:,idx, xs:xs+px, ys:ys+py])
                        canvas[:, :, idx, xs:xs+px, ys:ys+py] += out
            return canvas

        #else we proceed with the data as 2D
        x_steps, y_steps = get_steps_for_sliding_window(data.shape[2:], patch_size, overlap)

        for xs in x_steps:
            for ys in y_steps:
                    # check if out of bounds
                    out = self.forward(data[:, :, xs:xs+px, ys:ys+py])
                    canvas[:, :, xs:xs+px, ys:ys+py] += out
        return canvas


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
