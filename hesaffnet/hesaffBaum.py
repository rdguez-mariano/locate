#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

from copy import deepcopy

from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ellT, abc2A
from Utils import line_prepender
from architectures import AffNetFast
from HandCraftedModules import AffineShapeEstimator
USE_CUDA = False
try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
    nfeats = int(sys.argv[3])
except:
    print("Wrong input format. Try python hesaffBaum.py imgs/cat.png cat.txt 2000")
    sys.exit(1)

img = Image.open(input_img_fname).convert('RGB')
img = np.mean(np.array(img), axis = 2)

var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))

HA = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = nfeats, border = 5, num_Baum_iters = 16, AffNet = AffineShapeEstimator(patch_size=19))
if USE_CUDA:
    HA = HA.cuda()
    var_image_reshape = var_image_reshape.cuda()

LAFs, resp = HA(var_image_reshape)
ells  = LAFs2ellT(LAFs.cpu()).cpu().numpy()

np.savetxt(output_fname, ells, delimiter=' ', fmt='%10.10f')
line_prepender(output_fname, str(len(ells)))
line_prepender(output_fname, '1.0')
