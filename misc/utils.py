from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import misc.resnet as resnet
from misc.cnn_rnn import CnnRnn
import os
import random


def build_cnn(opt):
    net = getattr(resnet, opt.cnn_model)()
    if vars(opt).get('cnn_weight', '') != '':
        net.load_state_dict(torch.load(opt.cnn_weight))
    net = nn.Sequential(\
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool,
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4)
    if vars(opt).get('start_from', None) is not None \
            and os.path.isfile(os.path.join(opt.start_from, 'model-cnn.pth')) :
        if vars(opt).get('start_from_best', 0) :
            net.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-cnn-best.pth')))
            print('cnn load from %s'%(os.path.join(opt.start_from, 'model-cnn-best.pth')))
        else :
            net.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-cnn.pth')))
            print('cnn load from %s'%(os.path.join(opt.start_from, 'model-cnn.pth')))
    return net

def build_cnnrnn(cnn, rnn, opt) :
    cnnrnn = CnnRnn(cnn, rnn, opt.seq_per_img, opt.use_fc)
    return cnnrnn


def prepro_images(imgs, cnn_input_size = 224, data_augment=False):
    # crop the image
    h,w = imgs.shape[2], imgs.shape[3]

    # cropping data augmentation, if needed
    if h > cnn_input_size or w > cnn_input_size:
        if data_augment:
          xoff, yoff = random.randint(0, w-cnn_input_size), random.randint(0, h-cnn_input_size)
        else:
          # sample the center
          xoff, yoff = (w-cnn_input_size)//2, (h-cnn_input_size)//2
    # crop.
    imgs = imgs[:,:, yoff:yoff+cnn_input_size, xoff:xoff+cnn_input_size]

    return imgs

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'show_tell_t', 'all_img', 'fc']:
        return False
    return True

def if_use_fixatt(caption_model):
    if caption_model in ['fix_att_raw_seq']:
        return True
    return False

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq,invert=0):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if invert==0:
                    if j >= 1:
                        txt = txt + ' '
                    txt = txt + ix_to_word[str(ix.cpu().numpy())]
                else:
                    if j >= 1:
                        txt = ' '+txt
                    txt = ix_to_word[str(ix.cpu().numpy())]+txt
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        #output = - input * reward * Variable(mask)
        #output = torch.sum(output) / torch.sum(mask)
        output = - input * reward #* Variable(mask)
        output = torch.sum(output) / torch.sum(mask)
        return output

class RewardCriterion_WC(nn.Module):
    def __init__(self):
        super(RewardCriterion_WC, self).__init__()

    def forward(self, input, seq, sam_logprobs, reward):
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        #output = - input * reward * Variable(mask)
        #output = torch.sum(output) / torch.sum(mask)

        output = - torch.exp(input) * reward #* Variable(mask)
        output=torch.sum(output.view(output.size(0),-1),1)

        output = torch.sum(output) / torch.sum(mask)
        return output

class RewardCriterion_SCST_with_WC(nn.Module):
    def __init__(self):
        super(RewardCriterion_SCST_with_WC, self).__init__()

    def forward(self, input, seq, reward):
        mask = (seq>0).float()
        #mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        #output = - input * reward * Variable(mask)
        #output = torch.sum(output) / torch.sum(mask)
        output = - input * reward #* Variable(mask)
        output = torch.sum(output) / (torch.sum(mask)+mask.size(0))
        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class ActionCriterion(nn.Module):
    def __init__(self):   
        super(ActionCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

