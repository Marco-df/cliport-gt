import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.utils.utils as utils
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.core.clip import build_model, load_clip, tokenize

from cliport.models.core import fusion
from cliport.models.core.fusion import FusionConvLat


class CLIPDetector(nn.Module):
    """ CLIP-based success detector from images """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super(CLIPDetector, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim

        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.preprocess = preprocess

        self._load_clip()
        self._build_classifier()


    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)
        del model

    def _build_classifier(self):
        # language

        self.layer1 = nn.Linear(4096, 2048)
        self.layer2 = nn.Linear(2048, 1)
        
        self.pool = nn.AvgPool2d(7)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()


    def encode_image(self, img):
        with torch.no_grad():
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize([x]).to(self.device)
            text_enc, _ = self.clip_rn50.encode_text_with_embeddings(tokens)

        #text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_enc

    def forward(self, x1, x2, lat, l):
        x = self.preprocess(x, dist='clip')

        in_type = x1.dtype
        in_shape = x1.shape
        x1 = x1[:,:3]  # select RGB
        x2 = x2[:,:3] 
        x1, _ = self.encode_image(x1)
        x2, _ = self.encode_image(x2)
        x1 = x1.to(in_type)
        x2 = x2.to(in_type)

        text_enc = self.encode_text(l)
        text_enc = text_enc.to(dtype=x.dtype)

        #assert x1.shape[1] == self.input_dim
        
        x = torch.cat((x1, x2), dim=1)
        x = self.pool(x)
        x = torch.flatten(x)
        x = self.layer1(x)
        x = self.activation1(x)
        
        x = torch.cat((x, text_enc), dim=1)
        x = self.layer2(x)
        x = self.activation2(x)

        return x