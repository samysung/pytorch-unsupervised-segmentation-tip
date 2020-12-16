#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
import rasterio
from net import MyNet
from util import load_raster, write_raster

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False,
                    help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float,
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float,
                    help='step size for scribble loss')
args = parser.parse_args()

# load image
im, profile = load_raster(args.input)
print(im.shape)

data = torch.from_numpy(np.array([im.transpose((0, 2, 1)).astype('float32')/255.]))
if use_cuda:
    data = data.cuda()
data = Variable(data)

# load scribble
if args.scribble:

    mask, _ = load_raster(args.input.replace('.' + args.input.split('.')[-1], '_scribble.tif'), -1)
    mask = mask.reshape(-1)
    mask_inds = np.unique(mask)
    mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
    inds_sim = torch.from_numpy(np.where(mask == 255)[0])
    inds_scr = torch.from_numpy(np.where(mask != 255)[0])
    target_scr = torch.from_numpy(mask.astype(np.int))

    if use_cuda:
        inds_sim = inds_sim.cuda()
        inds_scr = inds_scr.cuda()
        target_scr = target_scr.cuda()

    target_scr = Variable(target_scr)
    # set minLabels
    args.minLabels = len(mask_inds)

# train
model = MyNet(profile["count"], args.nChannel, args.nConv)

if use_cuda:

    model.cuda()

model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# scribble loss definition
loss_fn_scr = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average=True)
loss_hpz = torch.nn.L1Loss(size_average=True)

HPy_target = torch.zeros(im.shape[1]-1, im.shape[2], args.nChannel)
HPz_target = torch.zeros(im.shape[1], im.shape[2]-1, args.nChannel)

if use_cuda:

    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))

for batch_idx in range(args.maxIter):

    # forwarding
    optimizer.zero_grad()
    output = model(data)[0]
    print(output.shape)
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

    outputHP = output.reshape((im.shape[1], im.shape[2], args.nChannel))
    HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
    HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
    lhpy = loss_hpy(HPy, HPy_target)
    lhpz = loss_hpz(HPz, HPz_target)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))

    if args.visualize:

        im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
        rgb_shape = [im.shape[1], im.shape[2], 3]
        im_target_rgb = im_target_rgb.reshape(rgb_shape).astype(np.uint8)
        cv2.imshow("output", im_target_rgb)
        cv2.waitKey(10)

    # loss
    if args.scribble:

        loss = args.stepsize_sim * loss_fn(output[inds_sim], target[inds_sim])
        + args.stepsize_scr * loss_fn_scr(output[inds_scr], target_scr[inds_scr])
        + args.stepsize_con * (lhpy + lhpz)

    else:

        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)

    loss.backward()
    optimizer.step()

    print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

    if nLabels <= args.minLabels:

        print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

# save output image
if not args.visualize:

    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

write_raster(im_target_rgb, "output.tif", profile)
