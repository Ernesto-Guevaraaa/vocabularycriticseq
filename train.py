# Use tensorboard

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from misc.AdaRandom import AdaRandom

import numpy as np
import time
import os
from six.moves import cPickle
import time
import threading
from threading import Thread
import sys
from queue import Empty, Full, Queue
import gc

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils


try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


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
        for batch, data in enumerate(dataset_generator):
            data['labels'] = np.zeros([opt.batch_size * opt.seq_per_img, opt.seq_length + 2], dtype='int64')
            data['masks'] = np.zeros([opt.batch_size * opt.seq_per_img, opt.seq_length + 2], dtype='float32')
            for i in range(opt.batch_size):
                data['labels'][i * opt.seq_per_img: (i + 1) * opt.seq_per_img, 1: opt.seq_length + 1] = data['seq_list'][i]
            nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
            for ix, row in enumerate(data['masks']):
                row[:nonzeros[ix]] = 1
               
            # We fill the queue with new fetched batch until we reach the max       size.
            batches_queue.put((batch, data), block=True)
            if tokill() == True:
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue, opt):
    """Thread worker for transferring pytorch tensors into
    GPU. batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while tokill() == False:
        batch, data = batches_queue.get(block=True)

        tmp = [data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        data['labels'], data['masks'] = tmp

        if opt.use_img:
            data['img'] = utils.prepro_images(np.stack(data['img']), opt.img_csize, True)
            data['img'] = Variable(torch.from_numpy(data['img']), requires_grad=False).cuda()
        else:
            data['fc_feats'] = np.stack(data['fc_feats'])
            data['fc_feats'] = Variable(torch.from_numpy(data['fc_feats']), requires_grad=False).cuda()
            if opt.use_att:
                data['att_feats'] = np.stack(data['att_feats'])
                data['att_feats'] = Variable(torch.from_numpy(data['att_feats']), requires_grad=False).cuda()

        cuda_batches_queue.put((batch, data), block=True)
        
        if tokill() == True:
            return


def train(opt):
    # Our train batches queue can hold at max 12 batches at any given time.
    # Once the queue is filled the queue is locked.
    train_batches_queue = Queue(maxsize=12)
    # Our numpy batches cuda transferer queue.
    # Once the queue is filled the queue is locked
    # We set maxsize to 3 due to GPU memory size limitations
    cuda_batches_queue = Queue(maxsize=5)

    training_set_generator = InputGen(opt, 'train')
    opt.use_att = utils.if_use_att(opt.caption_model)
    opt.vocab_size = training_set_generator.vocab_size
    opt.seq_length = training_set_generator.seq_length

    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)
    preprocess_workers = 4

    # We launch 4 threads to do load &amp;&amp; pre-process the input images
    for _ in range(preprocess_workers):
        t = Thread(target=threaded_batches_feeder, \
                   args=(train_thread_killer, train_batches_queue, training_set_generator))
        t.start()
    
    cuda_transfers_thread_killer = thread_killer()
    cuda_transfers_thread_killer.set_tokill(False)
    
    cu_preprocess_workers = 1
    for _ in range(cu_preprocess_workers):
        cudathread = Thread(target=threaded_cuda_batches, \
                       args=(cuda_transfers_thread_killer, cuda_batches_queue, train_batches_queue, opt))
        cudathread.start()

    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    infos = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.old_id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

    iteration = infos.get('iter', 0)
    start_epoch = infos.get('epoch', 0)

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    else:
        best_val_score=None
        
    model = models.setup(opt)
    model.cuda()
    if opt.use_img != 0 :
        cnn_model = utils.build_cnn(opt)
        cnn_model.cuda()
        cnnrnn_model = utils.build_cnnrnn(cnn_model, model, opt)
    else :
        cnn_model = None
        cnn_model_ = None
        cnnrnn_model = None

    lang_crit = utils.LanguageModelCriterion()

    if opt.optim == 'adam' :
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    elif opt.optim == 'sgd' :
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum)
    elif opt.optim == 'adarandom' :
        optimizer = AdaRandom(model.parameters(), lr=opt.learning_rate)
    else :
        print('this optim is not supported...')
    optimizer.zero_grad()
    if opt.use_img != 0 and opt.finetune_cnn_after != -1 :
        # only finetune the layer2 to layer4
        if opt.optim == 'adam' :
            cnn_optimizer = optim.Adam([\
                {'params': module.parameters()} for module in cnn_model._modules.values()[5:]\
                ], lr=opt.cnn_learning_rate, weight_decay=opt.cnn_weight_decay)
        elif opt.optim == 'SGD' :
            cnn_optimizer = optim.SGD([\
                    {'params': module.parameters()} for module in cnn_model._modules.values()[5:]\
                    ], lr=opt.cnn_learning_rate, weight_decay=opt.cnn_weight_decay, momentum=opt.momentum)
        cnn_optimizer.zero_grad()
        if opt.gpu_num > 1:
            cnnrnn_model = torch.nn.DataParallel(cnnrnn_model)
    else :
        if opt.gpu_num > 1:
            model_ = torch.nn.DataParallel(model)
        else :
            model_ = model

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        if os.path.isfile(os.path.join(opt.start_from, 'optimizer.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
        if opt.use_img != 0 and opt.finetune_cnn_after != -1 :
            if os.path.isfile(os.path.join(opt.start_from, 'optimizer-cnn.pth')):
                cnn_optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer-cnn.pth')))

    # Assure in training mode
    model.train()


    # We let queue to get filled before we start the training
    time.sleep(5)
    batches_per_epoch = training_set_generator.get_samples_count() // opt.batch_size
    for epoch in range(start_epoch, opt.max_epochs + 1):
        # eval model
        #opt.current_lr = opt.learning_rate
        #utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
        #if False:
        if True or epoch % opt.save_every == 0 :
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            lang_val_loss, act_val_loss, predictions, lang_stats = eval_utils.eval_split(cnn_model, model, lang_crit, opt, eval_kwargs)

            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation lang loss', lang_val_loss, iteration)
                #add_summary_value(tf_summary_writer, 'validation act loss', act_val_loss, iteration)
                if opt.language_eval == 1:
                    for k,v in lang_stats.items():
                        add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - lang_val_loss

            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            if opt.save_every and epoch % opt.save_every == 0:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-epoch%d.pth'%(epoch))
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer-epoch%d.pth'%(epoch))
                torch.save(optimizer.state_dict(), optimizer_path)

            if opt.use_img != 0 :
                cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn.pth')
                torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                if opt.save_every  and epoch % opt.save_every == 0:
                    cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn-epoch%d.pth'%(epoch))
                    torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                print("cnn model saved to {}".format(cnn_checkpoint_path))
                if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
                    cnn_optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer-cnn.pth')
                    torch.save(cnn_optimizer.state_dict(), cnn_optimizer_path)
                    if opt.save_every  and epoch % opt.save_every == 0:
                        cnn_optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer-cnn-epoch%d.pth'%(epoch))
                        torch.save(cnn_optimizer.state_dict(), cnn_optimizer_path)


            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = training_set_generator.get_vocab()

            with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            if opt.save_every  and epoch % opt.save_every == 0:
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-%d'%(epoch)+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)

            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                if opt.use_img != 0 :
                    cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn-best.pth')
                    torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                    print("cnn model saved to {}".format(cnn_checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)

            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            # Assign
            if opt.cut_seq_p > 0 and opt.cut_seq_p < 1 and epoch > opt.cut_seq_p_start and epoch % opt.cut_seq_p_incre_every == 0:
                opt.cut_seq_p += opt.cut_seq_p_incre_rate
            # Update the training stage of cnn
            if opt.use_img != 0 :
                if opt.finetune_cnn_after == -1 or epoch < opt.finetune_cnn_after:
                    for p in cnn_model.parameters():
                        p.requires_grad = False
                    cnn_model.eval()
                    print('cnn finetune is off')
                else:
                    for p in cnn_model.parameters():
                        p.requires_grad = True
                    # Fix the first few layers:
                    for module in cnn_model._modules.values()[:5]:
                        for p in module.parameters():
                            p.requires_grad = False
                    cnn_model.train()
                    print('cnn finetune is on')
            if opt.fix_rnn != 0 :
                for p in model.parameters() :
                    p.requires_grad = False
                model.eval()
                print('rnn finetune is off')
            else :
                for p in model.parameters() :
                    p.requires_grad = True
                model.train()
                print('rnn finetune is on')
            update_lr_flag = False
            print('rnn learning rate: %f'%(opt.current_lr))
            print('scheduled sampling prob: %f'%(model.ss_prob))
            #print('cut_seq_p prob: %f'%(opt.cut_seq_p))

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs:
            break

        for batch in range(batches_per_epoch):
            # We fetch a GPU batch in 0's due to the queue mechanism
            torch.cuda.synchronize()
            start = time.time()
            _, data = cuda_batches_queue.get(block=True)
            torch.cuda.synchronize()
            read_data_time = time.time() - start

            start = time.time()
            images = None
            fc_feats = None
            att_feats = None
            topics = None

            if opt.use_img:
                images = data['img']
            else:
                fc_feats, att_feats = data['fc_feats'], data['att_feats']
    
            labels,  masks = data['labels'], data['masks']


            if opt.use_img != 0 :
                seq_output = cnnrnn_model(images, labels)

                lang_loss = lang_crit(seq_output, labels[:, 1:], masks[:, 1:])
                loss = lang_loss
            else :
                seq_output = model_(fc_feats, att_feats, labels)

                lang_loss = lang_crit(seq_output, labels[:,1:], masks[:,1:])
                loss = lang_loss

            loss_ = loss / opt.iter_times
            loss_.backward()
            if (iteration+1) % opt.iter_times == 0:
                if opt.fix_rnn == 0 :
                    utils.clip_gradient(optimizer, opt.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                if opt.use_img != 0 and opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
                    utils.clip_gradient(cnn_optimizer, opt.grad_clip)
                    cnn_optimizer.step()
                    cnn_optimizer.zero_grad()
            train_lang_loss = lang_loss.data[0]
            #train_act_loss = act_loss.data[0]
            torch.cuda.synchronize()
            end = time.time()
            if iteration % 25 == 0 :
                print("iter {} (epoch {}), train_lang_loss = {:.3f}, time/batch = {:.3f}, time2/batch = {:.3f}" \
                    .format(iteration, epoch, train_lang_loss, end - start, read_data_time))

            # Update the iteration and epoch
            iteration += 1
            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                if tf is not None:
                    add_summary_value(tf_summary_writer, 'train_lang_loss', train_lang_loss, iteration)
                    #add_summary_value(tf_summary_writer, 'train_act_loss', train_act_loss, iteration)
                    add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
                    add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                    tf_summary_writer.flush()
            
            gc.collect()
    train_thread_killer.set_tokill(True)
    cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(preprocess_workers):
        try:
            # Enforcing thread shutdown
            train_batches_queue.get(block=True, timeout=1)
            cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass


opt = opts.parse_opt()
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
train(opt)
