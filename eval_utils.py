from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import os
import sys
import threading
from threading import Thread
from queue import Empty, Full, Queue
import misc.utils as utils
import gc
import math

from dataloader import *
import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import opts
import models
import misc.utils as utils



def language_eval(dataset, preds, model_id, split,annfile):
    annFile = annfile
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('/data3/liuhan/WCST/eval_results'):
        os.mkdir('/data3/liuhan/WCST/eval_results')
    cache_path_pred = os.path.join('/data3/liuhan/WCST/eval_results/', model_id + '_' + split + '_pred.json')
    cache_path = os.path.join('/data3/liuhan/WCST/eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path_pred, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path_pred)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    #cocoEval.params['image_id'] = cocoRes.imgToAnns.keys()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    """Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    while tokill() == False:
        for batch, data \
                in enumerate(dataset_generator):
            # We fill the queue with new fetched batch until we reach the max       size.
            batches_queue.put((batch, data) \
                              , block=True)
            if tokill() == True:
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue, opt):
    """Thread worker for transferring pytorch tensors into
    GPU. batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while tokill() == False:
        batch, data = batches_queue.get(block=True)
        if opt.use_label:
          data['labels'] = np.zeros([opt.batch_size * opt.seq_per_img, opt.seq_length + 2], dtype='int64')
          data['masks'] = np.zeros([opt.batch_size * opt.seq_per_img, opt.seq_length + 2], dtype='float32')
          for i in range(opt.batch_size):
              data['labels'][i * opt.seq_per_img: (i + 1) * opt.seq_per_img, 1: opt.seq_length + 1] = data['seq_list'][i]
          nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
          for ix, row in enumerate(data['masks']):
              row[:nonzeros[ix]] = 1

        if opt.use_img:
            data['img'] = utils.prepro_images(np.stack(data['img']), opt.img_csize, True)
        else:
            data['fc_feats'] = np.stack(data['fc_feats'])
            if opt.use_att:
                data['att_feats'] = np.stack(data['att_feats'])

        cuda_batches_queue.put((batch, data), block=True)
        if tokill() == True:
            return


def eval_split(cnn_model, model, lang_crit, opt, eval_kwargs={},dataloader=None):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_img = eval_kwargs.get('use_img', 0)
    img_csize = eval_kwargs.get('img_csize', 224)
    use_fc = eval_kwargs.get('use_fc', 0)
    use_att = eval_kwargs.get('use_att', 0)
    gpu_num = eval_kwargs.get('gpu_num', 1)
    invert=eval_kwargs.get('invert', 0)
    # Make sure in the evaluation mode
    if use_img != 0 :
        cnn_model.eval()
    model.eval()

    test_batches_queue = Queue(maxsize=12)
    # Our numpy batches cuda transferer queue.
    # Once the queue is filled the queue is locked
    # We set maxsize to 3 due to GPU memory size limitations
    cuda_batches_queue = Queue(maxsize=3)
    if dataloader:
        test_set_generator=dataloader
    else:
        test_set_generator = InputGen(opt, split)
    opt.use_att = utils.if_use_att(opt.caption_model)
    opt.vocab_size = test_set_generator.vocab_size
    try:
        opt.seq_length = test_set_generator.seq_length
    except:
        pass

    test_thread_killer = thread_killer()
    test_thread_killer.set_tokill(False)
    preprocess_workers = 1

    # We launch 4 threads to do load &amp;&amp; pre-process the input images
    for _ in range(preprocess_workers):
        t = Thread(target=threaded_batches_feeder, \
                   args=(test_thread_killer, test_batches_queue, test_set_generator))
        t.start()
    
    cuda_transfers_thread_killer = thread_killer()
    cuda_transfers_thread_killer.set_tokill(False)
    cudathread = Thread(target=threaded_cuda_batches, \
                        args=(cuda_transfers_thread_killer, cuda_batches_queue, test_batches_queue, opt))
    cudathread.start()



    n = 0
    loss = 0
    lang_loss_sum = 0
    act_loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    #done_beam = []
    #image_ids = []
    if num_images<0:
        num_images=test_set_generator.get_samples_count()
    #batches_per_epoch = num_images // opt.batch_size
    batches_per_epoch = int(math.ceil(num_images / opt.batch_size))
    if num_images % opt.batch_size != 0:
        batches_per_epoch += 1
    #att_h = np.zeros((num_images * 5, 25, 512))
    #lang_h = np.zeros((num_images * 5, 25, 512))
    for batch in range(batches_per_epoch):
        # Load data from train split (0)
        _, data = cuda_batches_queue.get(block=True)
        images = None
        fc_feats = None
        att_feats = None
        
        if opt.use_label:
            labels = Variable(torch.from_numpy(data['labels']), volatile=True).cuda()
            masks = Variable(torch.from_numpy(data['masks']), volatile=True).cuda()

        if opt.use_img:
            images = data['img']
            images = Variable(torch.from_numpy(images), volatile=True).cuda()
            att_feats = cnn_model(images).permute(0, 2, 3, 1)
            fc_feats = att_feats.mean(2).mean(1)
            att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), test_set_generator.seq_per_img,) \
                                    + att_feats.size()[1:])).contiguous().view(
                                    *((att_feats.size(0) * test_set_generator.seq_per_img,) + att_feats.size()[1:]))
            fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), test_set_generator.seq_per_img,) \
                                      + fc_feats.size()[1:])).contiguous().view(
                                        *((fc_feats.size(0) * test_set_generator.seq_per_img,) + fc_feats.size()[1:]))
        else:
            fc_feats, att_feats = data['fc_feats'], data['att_feats']
            tmp = [fc_feats, att_feats]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats = tmp

        # forward the model to get loss
        if data.get('labels', None) is not None:
            seq_output = model(fc_feats, att_feats, labels)

            lang_loss = lang_crit(seq_output, labels[:,1:], masks[:,1:]).cpu().data.numpy()
            lang_loss_sum = lang_loss_sum + lang_loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        att_feats = att_feats.data.cpu().numpy()[np.arange(test_set_generator.batch_size) * test_set_generator.seq_per_img]
        att_feats = Variable(torch.from_numpy(att_feats), volatile=True).cuda()
        if use_fc != 0 :
            fc_feats = fc_feats.data.cpu().numpy()[np.arange(test_set_generator.batch_size) * test_set_generator.seq_per_img]
            fc_feats = Variable(torch.from_numpy(fc_feats), volatile=True).cuda()
        model.sample(fc_feats, att_feats, eval_kwargs)
        try:
            seq, _ = model.sample(fc_feats, att_feats,eval_kwargs)
        except:
            seq, _ = model.module.sample(fc_feats, att_feats, eval_kwargs)

        #set_trace()
        sents = utils.decode_sequence(test_set_generator.get_vocab(), seq,invert=invert)

        for k, sent in enumerate(sents):
            entry = {'caption': sent, 'image_id': data['infos'][k]['id']}
            #image_ids.append(data['infos'][k]['image_id'])
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['image_id']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(batch, batches_per_epoch, loss))
        gc.collect()

    test_thread_killer.set_tokill(True)
    cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(preprocess_workers):
        try:
            # Enforcing thread shutdown
            test_batches_queue.get(block=True, timeout=1)
            cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass
   
    pred_len = len(predictions)
    if pred_len > num_images:
        for i in range(pred_len - num_images):
            predictions.pop()
    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split,opt.annFile)

    # Switch back to training mode
    if use_img != 0 :
        cnn_model.train()
    model.train()
    gc.collect()
    #return lang_loss_sum/loss_evals, act_loss_sum/loss_evals, predictions, lang_stats
    return lang_loss_sum/loss_evals, None, predictions, lang_stats


if __name__ == "__main__":
    opt = opts.parse_opt()
    opt.use_att = utils.if_use_att(opt.caption_model)
    opt.vocab_size = 9487
    opt.seq_length = 16
    model = models.setup(opt)
    model.cuda()
    lang_crit = utils.LanguageModelCriterion()
    if opt.use_img != 0 :
        cnn_model = utils.build_cnn(opt)
        cnn_model.cuda()
    else :
        cnn_model = None

    eval_kwargs = {'split': opt.test_split}
    eval_kwargs.update(vars(opt))
    lang_val_loss, act_val_loss, predictions, lang_stats = eval_split(cnn_model, model, lang_crit, opt, eval_kwargs)
    json.dump(predictions, open('/data3/liuhan/WCST/predictions/%s.json'%(opt.id), 'w'), indent=2, ensure_ascii=False, encoding="utf-8")




