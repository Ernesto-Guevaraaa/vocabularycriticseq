from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import threading

import torch

from scipy.misc import imread, imresize
import multiprocessing
from torchvision import transforms as trn

preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_img_data(file_path, img_size):
    img = imread(file_path)
    img = imresize(img, (img_size, img_size))

    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

    img = preprocess(torch.from_numpy(img.transpose(2, 0, 1).astype('float32') / 255.0)).numpy()
    return img

def get_npy_data(ix, fc_file, att_file, use_att,att_num=-1):
    if use_att == True:
        if att_num>0:
            fc_data=np.load(fc_file)
            try:
                att_data_raw=np.load(att_file)['feat']
            except:
                att_data_raw=np.load(att_file)
                 
            att_shape=[att_num,len(att_data_raw[0])]
            att_data=np.zeros(att_shape)
            att_data[:len(att_data_raw),:]=att_data_raw
            att_data=np.float32(att_data)
        else:
            fc_data=np.load(fc_file)
            try:
                att_data=np.load(att_file)['feat']
            except:
                att_data=np.load(att_file)
            return [fc_data,att_data]
    else:
        fc_data=np.load(fc_file)
        att_data=np.zeros((1, 1, 1))

    return [fc_data[0],att_data]
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def get_path_i(paths_count):
    """Cyclic generator of paths indice
    """
    current_path_id = 0
    while True:
        yield current_path_id
        current_path_id = (current_path_id + 1) % paths_count


class InputGen:
    def __init__(self, opt, split):
        #self.paths = paths
        self.index = 0
        #self.batch_size = batch_size
        self.init_count = 1
        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.images, self.seq_list, self.topics, self.fcs, self.atts, self.infos, self.gts \
            = [], [], [], [], [], [], []

        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.use_att = getattr(opt, 'use_att', True)
        self.use_img = getattr(opt, 'use_img', 1)
        self.img_fold = getattr(opt, 'img_fold', 'data/images')
        self.img_size = getattr(opt, 'img_size', 256)
        self.use_fc = getattr(opt, 'use_fc', 1)
        self.use_topic = getattr(opt, 'use_topic', 1)
        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.use_detection_att=getattr(opt, 'use_detection_att', 1)
        self.use_label=getattr(opt, 'use_label', 1)
        self.use_new_data=getattr(opt, 'use_new_data', 0)
        self.invert=getattr(opt, 'invert', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        # open the label file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_label_h5, opt.input_topic_h5)
        
        if self.use_label!=0:
          self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
          # open topic file
          if self.use_topic != 0 :
              self.h5_topic_file = h5py.File(self.opt.input_topic_h5,'r')
  
          # load in the sequence data
          seq_size = self.h5_label_file['labels'].shape
          self.seq_length = seq_size[1]
          print('max sequence length in data is', self.seq_length)
          # load the pointers in full to RAM (should be small enough)
          self.label_start_ix = self.h5_label_file['label_start_ix'][:]
          self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        # separate out indexes for each of the provided splits
        if self.use_new_data==0:
            self.split_ix = {'train': [], 'val': [], 'test': []}
            if opt.dataset_ix_start >= 0:
                ix_list = range(opt.dataset_ix_start, opt.dataset_ix_end)
            else:
                ix_list = range(len(self.info['images']))
            for ix in ix_list:
                img = self.info['images'][ix]
                if img['split'] == 'train':
                    self.split_ix['train'].append(ix)
                elif img['split'] == 'val':
                    self.split_ix['val'].append(ix)
                elif img['split'] == 'test':
                    self.split_ix['test'].append(ix)
    
            print('assigned %d images to split train' % len(self.split_ix['train']))
            print('assigned %d images to split val' % len(self.split_ix['val']))
            print('assigned %d images to split test' % len(self.split_ix['test']))
            if split == 'all':
                self.ix_list = ix_list
            else:
                self.ix_list = self.split_ix[split]
        else:
            self.info={}
            self.info['images']=[]
            self.info['ix_to_word']=self.ix_to_word 
            for image in os.listdir(opt.img_fold):
                if image.split('.')[-1]=='jpg':
                    image_id=image.split('.')[-2]
                    image_info={}
                    image_info['image_id']=image_id
                    image_info['id']=image_id
                    self.info['images'].append(image_info)
                    
            self.ix_list=range(len(self.info['images']))

        self.path_id_generator = threadsafe_iter(get_path_i(len(self.ix_list)))
        
    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_samples_count(self):
        """ Returns the total number of images needed to train an epoch """
        return len(self.ix_list)

    def get_batches_count(self):
        """ Returns the total number of batches needed to train an epoch """
        return int(self.get_samples_count() / self.batch_size)

    def pre_process_input(self, im, lb):
        """ Do your pre-processing here
                Need to be thread-safe function"""
        return im, lb

    def next(self):
        return self.__iter__()

    def __iter__(self):
        while True:
            # In the start of each epoch we shuffle the data paths
            with self.lock:
                if (self.init_count == 0):
                    random.shuffle(self.ix_list)
                    self.images, self.seq_list,  self.fcs, self.atts, self.topics, self.gts, self.infos \
                            = [], [], [], [], [], [], []
                    self.init_count = 1
            # Iterates through the input paths in a thread-safe manner
            for ix_i in self.path_id_generator:
                # fetch image
                ix = self.ix_list[ix_i]
                if self.use_img != 0:
                    # raw_image = self.h5_image_file['images'][ix, :, :, :]
                    # img = preprocess(torch.from_numpy(raw_image.astype('float32')/255.0)).numpy()
                    img = get_img_data(os.path.join(self.img_fold, str(self.info['images'][ix]['image_id']) + '.jpg'),\
                            self.img_size)
                else:
                    fc_file=os.path.join(self.input_fc_dir, str(self.info['images'][ix]['image_id']) + '.npy')
                    att_file=os.path.join(self.input_att_dir, str(self.info['images'][ix]['image_id']) + '.npz')
                    if not os.path.exists(att_file):
                        att_file=os.path.join(self.input_att_dir, str(self.info['images'][ix]['image_id']) + '.npy')
                    fc, att = get_npy_data(ix, \
                                fc_file,
                                att_file,
                                self.use_att,
                                self.opt.att_feat_num
                                )

                if self.use_topic != 0:
                    topic = self.h5_topic_file['topics'][ix, :]
                
                if self.use_label!=0:
                  # fetch the sequence labels
                  ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
                  ix2 = self.label_end_ix[ix] - 1
                  ncap = ix2 - ix1 + 1  # number of captions available for this image
                  assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'
                  if ncap < self.seq_per_img:
                      # we need to subsample (with replacement)
                      seq = np.zeros([self.seq_per_img, self.seq_length], dtype='int')
                      for q in range(self.seq_per_img):
                          ixl = random.randint(ix1, ix2)
                          seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
  
                  else:
                      ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
                      seq = self.h5_label_file['labels'][ixl: ixl + self.seq_per_img, :self.seq_length]
                      
                  if self.invert!=0:
                      for i in range(self.seq_per_img):
                          seq_=np.zeros([self.seq_length])
                          length=np.sum(seq[i]!=0)
                          for j in range(length):
                              seq_[j]=seq[i][length-j-1]
                          seq[i]=seq_
                          
  
                  # Used for reward evaluation
                  gt = self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]]
                else:
                  seq=None
                  
                # record associated info as well
                info_dict = {}
                info_dict['image_id'] = self.info['images'][ix]['image_id']
                info_dict['id'] = self.info['images'][ix]['id']

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if len(self.seq_list) < self.batch_size:
                        if self.use_img:
                            self.images.append(img)
                        else:
                            self.fcs += [fc] * self.seq_per_img
                            self.atts += [att] * self.seq_per_img
                        self.seq_list.append(seq)

                        if self.use_topic:
                            self.topics += [topic] * self.seq_per_img
                        
                        if self.use_label:
                            self.gts.append(gt)
                        self.infos.append(info_dict)
                    if len(self.seq_list) % self.batch_size == 0:
                        '''
                        self.labels = np.zeros([self.batch_size * self.seq_per_img, self.seq_length + 2], dtype='int')
                        self.masks = np.zeros([self.batch_size * self.seq_per_img, self.seq_length + 2], dtype='float32')
                        for i in range(self.batch_size):
                            self.labels[i * self.seq_per_img: (i + 1) * self.seq_per_img, 1: self.seq_length + 1] = self.seq_list[i]
                        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, self.labels)))
                        for ix, row in enumerate(self.masks):
                            row[:nonzeros[ix]] = 1
                        '''
                        data = {}
                        if self.use_img != 0:
                            #data['img'] = np.stack(self.images)
                            data['img'] = self.images
                        else:
                            data['fc_feats'] = self.fcs
                            data['att_feats'] = self.atts
                            #data['fc_feats'] = np.stack(self.fcs)
                            #data['att_feats'] = np.stack(self.atts)
                        if self.use_topic != 0:
                            #data['topics'] = np.stack(self.topics)
                            data['topics'] = self.topics
                        #data['labels'] = self.labels
                        data['seq_list'] = self.seq_list
                        data['gts'] = self.gts
                        #data['masks'] = self.masks
                        data['infos'] = self.infos
                        yield data
                        self.images, self.seq_list,self.fcs, self.atts, self.topics, self.gts, self.infos \
                            = [], [], [], [], [], [], []
            # At the end of an epoch we re-init data-structures
            with self.lock:
                self.init_count = 0

    def __call__(self):
        return self.__iter__()







