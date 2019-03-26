from __future__ import division
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.contrib.nn import SyncBatchNorm
from gluoncv.model_zoo.segbase import SegBaseModel
from mxnet import as nd

# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F

from .base import BaseNet
from .fcn import FCNHead
from ..nn import Encoding, Mean


__all__ = ['EncNet', 'EncModule', 'get_encnet', 'get_encnet_resnet50_pcontext',
           'get_encnet_resnet101_pcontext', 'get_encnet_resnet50_ade',
           'get_encnet_resnet101_ade']

class EncNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True, lateral=False,
                 norm_layer=SyncBatchNorm, **kwargs):
        super(EncNet, self).__init__(nclass, backbone, aux, se_loss,
                                     norm_layer=norm_layer, **kwargs)
        self.head = EncHead(2048, self.nclass, se_loss=se_loss,
                            lateral=lateral, norm_layer=norm_layer,
                            up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def hybrid_forward(self, F，x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(*features))
        x[0] = F.contrib.BilinearResize2D(x[0], imsize, **self._up_kwargs)
#         if self.aux:
#             auxout = self.auxlayer(features[2])
#             auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
#             x.append(auxout)
        return tuple(x)


class EncModule(HybridBlock):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
#         self.encoding = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 1, bias=False),
#             norm_layer(in_channels),
#             nn.ReLU(inplace=True),
#             Encoding(D=in_channels, K=ncodes),
#             norm_layer(ncodes),
#             nn.ReLU(inplace=True),
#             Mean(dim=1))
        with self.name_scope():
            self.encblock1 = nn.HybridSequential()
            self.encblock1.add(nn.Conv2D(in_channels, in_channels, 1, bias=False))
            self.encblock1.add(norm_layer(in_channels,
                              **({} if norm_kwargs is None else norm_kwargs)))
            self.relu = nn.Activation('relu')
            self.encblock2 = nn.HybridSequential()
            self.encblock2.add(Encoding(D=in_channels, K=ncodes))
            self.encblock2.add(norm_layer(ncodes,
                              **({} if norm_kwargs is None else norm_kwargs)))
            self.mean = Mean(dim=1)
            self.fcblock = nn.HybridSequential()
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels),
#             nn.Sigmoid())
            self.fcblock.add(nn.Linear(in_channels, in_channels))
            self.fcblock.add(nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def hybrid_forward(self, F, x):
        en = self.encblock1(x)
        en = self.relu(en)
        en = self.encblock2(en)
        en = self.relu(en)
        en = self.mean(en)
        b, c, _, _ = x.size()
        gamma = self.fcblock(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class EncHead(HybridBlock):
    def __init__(self, in_channels, out_channels, se_loss=True, lateral=True,
                 norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
#             norm_layer(512),
#             nn.ReLU(inplace=True))
        with self.name_scope()
            inter_channels = in_channels
            self.block = nn.HybridSequential(nn.Conv2d(in_channels, 512, 3, padding=1, bias=False))
            self.block.add(norm_layer(512,
                      **({} if norm_kwargs is None else norm_kwargs)))
            self.relu = nn.Activation('relu')
            
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
#         self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
#                                    nn.Conv2d(512, out_channels, 1))
        self.dropout = nn.Dropout(0.1, False)
        self.classifier = nn.Conv2d(512, out_channels, 1)

    def hybrid_forward(self, F,*inputs):
        feat = self.block(inputs[-1])
#         if self.lateral:
#             c2 = self.connect[0](inputs[1])
#             c3 = self.connect[1](inputs[2])
#             feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.dropout(outs[0])
        outs[0] = self.classifier(outs[0])
        return tuple(outs)


def get_encnet(dataset='citys', backbone='resnet50', pretrained=False,
               root='~/.encoding/models', **kwargs):
    r"""EncNet model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    backbone : str, default resnet50
        The backbone network. (resnet50, 101, 152)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    kwargs['lateral'] = True if dataset.lower().startswith('p') else False
    # infer number of classes
    from ..datasets import datasets, acronyms
    model = EncNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('encnet_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model