import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pickle as pkl
import argparse
import os
import json
import time

import misc.utils as utils
import models
import eval_utils

def predict_raw_data(opt):
    if opt.start_from_best:
        info_file=os.path.join(opt.start_from,'infos_'+opt.model_id+'-best.pkl')
    else:
        info_file = os.path.join(opt.start_from, 'infos_' + opt.model_id + '.pkl')
    with open(info_file,'r') as f:
        info=pkl.load(f)

    model_opt=info['opt']
    assert model_opt.id == opt.model_id, 'id mismatch'

    model_opt.start_from_best=opt.start_from_best
    if opt.caption_model !='':
        model_opt.caption_model=opt.caption_model
    if opt.input_json !='':
        model_opt.input_json=opt.input_json
    if opt.start_from!='':
        model_opt.start_from=opt.start_from
    model_opt.beam_size=opt.beam_size
    model_opt.sample_n=opt.sample_n
    model_opt.seq_per_img=1
    model_opt.use_img=0
    model_opt.batch_size=opt.batch_size
    model_opt.img_fold=opt.img_fold
    model_opt.att_feat_num=opt.att_feat_num
    model_opt.use_new_data=1
    model_opt.language_eval=0
    model_opt.input_fc_dir=opt.input_fc_dir
    model_opt.input_att_dir=opt.input_att_dir
    model_opt.use_label=0
    model_opt.val_images_use=-1

    if opt.cnn_weight !='':
        model_opt.cnn_weight=opt.cnn_weight

    model = models.setup(model_opt)
    if opt.gpu_num>0:
        model.cuda()
    if opt.gpu_num>1:
        model=torch.nn.DataParallel(model)

    model_opt.use_att = utils.if_use_att(model_opt.caption_model)
    model_opt.seq_length = 16

    eval_kwargs = {'sample_max': 1}
    eval_kwargs.update(vars(model_opt))
    _, _, predictions, _ = eval_utils.eval_split(None, model, None, model_opt, eval_kwargs)
    if opt.save_pred_file != '':
        save_file=opt.save_pred_file
    else:
        if not os.path.exists('predictions'):
            os.mkdir('predictions')
        save_file='predictions/%s.json' % (opt.pred_id+time.strftime('-%Y-%m-%d-%H-%M-%S',time.localtime()))
    json.dump(predictions, open(save_file, 'w'), indent=2, ensure_ascii=False, encoding="utf-8")



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--caption_model', type=str, default='')
    parser.add_argument('--input_json', type=str, help='json of MSCOCO2014', default='')
    parser.add_argument('--start_from',type=str,help='path of model params',default='')
    parser.add_argument('--model_id',type=str,help='the id of img-cap model')
    parser.add_argument('--pred_id', type=str, help='the id of prediction')
    parser.add_argument('--start_from_best',type=int,default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cnn_weight',type=str,default='')
    parser.add_argument('--img_fold',type=str)
    parser.add_argument('--save_pred_file',type=str,default='')
    parser.add_argument('--gpu_num', type=int,default=1,help='num of gpu')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--att_feat_num', type=int, default=100)
    parser.add_argument('--input_fc_dir', type=str)
    parser.add_argument('--input_att_dir', type=str)
    args=parser.parse_args()
    predict_raw_data(args)