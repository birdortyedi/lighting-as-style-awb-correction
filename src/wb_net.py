import torch.nn as nn
import torch
from torchvision.models import vgg16
from src import ifrnet


class WBnetIFRNet(nn.Module):
    def __init__(self, inchnls=9, initialchnls=16, ps=64, rows=4, columns=6, norm=False, device='cuda'):
        """ Network constructor.
    """
        self.outchnls = int(inchnls / 3)
        self.inchnls = inchnls
        self.device = device
        super(WBnetIFRNet, self).__init__()
        assert columns % 2 == 0, 'use even number of columns'
        assert columns > 1, 'use number of columns > 1'
        assert rows > 1, 'use number of rows > 1'
        # self.net = gridnet.network(inchnls=self.inchnls, outchnls=self.outchnls,
        #                    initialchnls=initialchnls, rows=rows, columns=columns,
        #                    norm=norm, device=self.device)

        vgg_feats = vgg16(pretrained=True).features.eval().cuda()
        vgg_feats = nn.Sequential(*[module for module in vgg_feats][:35]).eval()
        self.net = ifrnet.IFRNetv2(vgg_feats=vgg_feats, input_size=ps, base_n_channels=initialchnls, in_channels=self.inchnls, out_channels=self.outchnls).to(self.device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ Forward function"""
        weights = self.net(x)
        weights = torch.clamp(weights, -1000, 1000)
        weights = self.softmax(weights)
        out_img = torch.unsqueeze(weights[:, 0, :, :], dim=1) * x[:, :3, :, :]
        for i in range(1, int(x.shape[1] // 3)):
            out_img += torch.unsqueeze(weights[:, i, :, :],
                                       dim=1) * x[:, (i * 3):3 + (i * 3), :, :]
        return out_img, weights


if __name__ == '__main__':
    x = torch.rand(8, 15, 64, 64).cuda()
    net = WBnetIFRNet(15, 32)
    y, w = net(x)
    print(y.shape, w.shape)
