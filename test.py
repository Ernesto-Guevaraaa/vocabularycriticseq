# Use tensorboard

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import json

import time
import os
from six.moves import cPickle

import opts
import models
import eval_utils
import misc.utils as utils
from dataloader import *


def test(opt):
    split=getattr(opt, 'split', 'test')
    test_set_generator = InputGen(opt, split)
    opt.vocab_size = test_set_generator.vocab_size
    opt.seq_length = test_set_generator.seq_length
    opt.use_att = utils.if_use_att(opt.caption_model)
    #opt.vocab_size = 9487 
    #opt.seq_length = 16
    model = models.setup(opt)
    model.cuda()
    crit = utils.LanguageModelCriterion()
    if opt.use_img != 0 :
        cnn_model = utils.build_cnn(opt)
        cnn_model.cuda()
    else :
        cnn_model = None
    opt.id=opt.id+time.strftime('-%Y-%m-%d-%H-%M-%S',time.localtime())
    eval_kwargs = {'split': opt.test_split, 'sample_max': 1}
    eval_kwargs.update(vars(opt))
    loss, _, predictions, lang_state = eval_utils.eval_split(cnn_model, model, crit,opt, eval_kwargs,test_set_generator)
    json.dump(predictions, open('predictions/%s.json'%(opt.id), 'w'), indent=2, ensure_ascii=False, encoding="utf-8")

opt = opts.parse_opt()
test(opt)


