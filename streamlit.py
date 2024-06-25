import os

import torch
import torch.nn as nn
import streamlit as st
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

import torchvision.transforms as transforms 
import torchvision.utils as vutils

from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from IPython.display import display, clear_output



from PIL import Image
import numpy as np
import errno
import os
import pickle
import time
from glob import glob
import cv2
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import os
import pickle
import random
import time
import PIL
from PIL import Image



st.title("Pokemon image generator")
text_prompt = st.text_input("Enter a text prompt:")
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
text_prompt= text_prompt.lower()
embeddings=bert_model.encode(text_prompt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code
        output = self.outlogits(h_c_code)
        return output.view(-1)
def conv(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def up(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block

class residual(nn.Module):
    def __init__(self, channel_num):
        super(residual, self).__init__()
        self.block = nn.Sequential(
            conv(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out    
    
import torch
import torch.nn as nn

class CA_NET(nn.Module):
    def __init__(self, condition_dim=None, device=None):
        super(CA_NET, self).__init__()
        
        self.device = device if device is not None else torch.device('cpu')  # Set default device
        self.t_dim = 768
        self.c_dim = condition_dim if condition_dim is not None else 256  # Set default condition dimension
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True).to(self.device)
        self.relu = nn.ReLU().to(self.device)

    def encode(self, text_embedding):
        text_embedding = text_embedding.to(self.device)
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

class STAGE1_G(nn.Module):   #stage 1 generator 
    def __init__(self, df_dim, condition_dim):
        super(STAGE1_D, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = condition_dim
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = None

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding
    def __init__(self,
                 gf_dim, condition_dim, z_dim, device = device):
        
        super(STAGE1_G, self).__init__()
        self.gf_dim = gf_dim * 8
        self.ef_dim = condition_dim
        self.z_dim = z_dim
        self.define_module()

    def define_module(self):
        ninput = self.z_dim + self.ef_dim
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = up(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = up(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = up(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = up(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv(ngf // 16, 3),
            nn.Tanh())

    def forward(self, text_embedding, noise):
        c_code, mu, logvar = self.ca_net(text_embedding)
        z_c_code = torch.cat((noise, c_code), 1)
        h_code = self.fc(z_c_code)

        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        fake_img = self.img(h_code)
        return None, fake_img, mu, logvar
class STAGE1_D(nn.Module): #discriminator 
    def __init__(self, df_dim, condition_dim):
        super(STAGE1_D, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = condition_dim
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = None

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding

gf_dim = 128
condition_dim = 256
z_dim = 100
generator = STAGE1_G(gf_dim=gf_dim, condition_dim=condition_dim, z_dim=z_dim, device=device)
generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
generator.eval()
batch_size = 1
z_dim = 100
noise = torch.randn(batch_size, z_dim, dtype=torch.float32)
images, mu, logvar = generator(text_prompt, noise)
st.image(images, caption='Generated Image', use_column_width=True)
 




