import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .AttModel import *


def setup(opt):

    if opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    elif opt.caption_model == 'att2in':
        model = Att2in2Model(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None and opt.caption_model != 'ensemble':
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        #assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.old_id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        if opt.start_from_best :
            model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best.pth')))
            print('model load from %s'%(os.path.join(opt.start_from, 'model-best.pth')))
        else :
            model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
            print('model load from %s'%(os.path.join(opt.start_from, 'model.pth')))

    return model
