from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
from torch.autograd import Variable
import random

import sys

def init_scst_scorer(opt):
    print('preparing %s scorer'%(opt.tar_metric))
    if opt.tar_metric=='CIDEr':
        #sys.path.append("ciderD")
        from ciderD.pyciderevalcap.ciderD.ciderD import CiderD
        #scorer = CiderD(df='corpus')
        scorer = CiderD(df=opt.cached_tokens)

    elif opt.tar_metric=='Bleu':
        from pycocoevalcap.bleu.bleu import Bleu
        scorer=Bleu(n=4)
        
    elif opt.tar_metric=='Meteor':
        from pycocoevalcap.meteor.meteor import Meteor
        scorer=Meteor()
    
    elif opt.tar_metric=='Rouge':
        from pycocoevalcap.rouge.rouge import Rouge
        scorer=Rouge()
    return scorer
        

def array_to_str(arr,invert=0):
    out = ''
    if invert==0:
        for i in range(len(arr)):
            out += str(arr[i]) + ' '
            if arr[i] == 0:
                break
        return out.strip()
    else:
        for i in range(len(arr)):
            if arr[i] == 0:
                out = out + ' ' + str(arr[i]) 
                break
            out = ' ' + str(arr[i]) + out

        return out.strip()
    
def get_scst_reward(model, fc_feats, att_feats, inputs,data, gen_result,scorer,opt):
    invert=getattr(opt, 'invert', 0)
    
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    vocab_size=inputs.size(2) 

    # get greedy decoding baseline
    try:
        greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True),
                                            Variable(att_feats.data, volatile=True), None)
    except:
        greedy_res, _ = model.module.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True), None)

    res_ = []

    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res_.append([array_to_str(gen_result[i])])
    for i in range(batch_size):
        res_.append([array_to_str(greedy_res[i])])
        
    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j],invert) for j in range(len(data['gts'][i]))]

    #_, scores = Bleu(4).compute_score(gts, res)
    #scores = np.array(scores[3])
    res = [{'image_id':i, 'caption': res_[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.tar_metric=='CIDEr':
        _, scores = scorer.compute_score(gts, res)
    else:
        res={i: res_[i] for i in range(2 * batch_size)}
        _, scores = scorer.compute_score(gts, res)
        if opt.tar_metric=='Bleu':
            scores=scores[-1]
        scores=np.array(scores)
    #print('Cider scores:', _)

    base_score = np.mean(scores[batch_size:])
    explore_score = np.mean(scores[:batch_size] )
    scores = scores[:batch_size] - scores[batch_size:]
    
    rewards=np.zeros([batch_size,gen_result.shape[1],vocab_size])

    for i in range(batch_size):
        x=res_[i][0].split(' ')
        for j in range(len(x)):
            rewards[i][j][int(x[j])]=scores[i]

    return rewards, base_score, explore_score

'''  
def get_self_critical_reward(model, fc_feats, att_feats, data, gen_result):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    # get greedy decoding baseline
    try:
        greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True),
                                            Variable(att_feats.data, volatile=True), None)
    except:
        greedy_res, _ = model.module.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True), None)

    res = OrderedDict()

    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    #_, scores = Bleu(4).compute_score(gts, res)
    #scores = np.array(scores[3])
    res = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    #print('Cider scores:', _)
    base_score = np.mean(scores[batch_size:])
    expore_score = np.mean(scores[:batch_size] )
    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards, base_score, expore_score
'''
def get_wc_reward(model, fc_feats, att_feats,inputs,data,gen_result,scorer,opt):
    if opt.tar_metric=='CIDEr':
        return get_wc_reward_CIDEr_new(model, fc_feats, att_feats,inputs,data,gen_result,scorer,opt)
    elif opt.tar_metric=='Bleu':
        return get_wc_reward_Bleu(model, fc_feats, att_feats,inputs,data,gen_result,scorer,opt)

def get_wc_reward_CIDEr(model, fc_feats, att_feats,inputs,data,gen_result,scorer,opt):
    use_scst=getattr(opt, 'use_scst', 1)
    use_bos=getattr(opt, 'use_bos', 1)
    use_one_gram=getattr(opt, 'use_one_gram', 0)
    invert=getattr(opt, 'invert', 0)
    alpha=getattr(opt, 'current_alpha', 0)
    

    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    vocab_size=inputs.size(2) 

    gen_result = gen_result.cpu().numpy()

    res = []
    for i in range(batch_size):
        res.append([array_to_str(gen_result[i])])
    if use_scst!=0:
        greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True),
                                                Variable(att_feats.data, volatile=True))

        greedy_res = greedy_res.cpu().numpy()

        for i in range(batch_size):
            res.append([array_to_str(greedy_res[i])])

    gts = []
    for i in range(len(res)):
        i_=i % batch_size // seq_per_img
        data_gt=data['gts'][i_]
        
        if i<batch_size:
            #if len(data_gt)>5:
            #    index_list=list(range(5))
            #    random.shuffle(index_list)
            #    data_gt=[data_gt[t] for t in index_list]

            gts.append([array_to_str(data_gt[j],invert) for j in range(len(data_gt))])
        else:
            gts.append(gts[i%batch_size])

    scores,scores_dif_tc,scores_dif_l,keys_res=scorer.forward(res,gts)
    scores_dif_tc_=scores_dif_tc.copy()

    if use_scst!=0:
        reward_scst = scores[:batch_size] - scores[batch_size:]
        reward_scst_list=np.zeros([batch_size,gen_result.shape[1],vocab_size])
        base_score = np.mean(scores[batch_size:])
        explore_score = np.mean(scores[:batch_size])

        for i in range(batch_size):
            x=res[i][0].split(' ')
            for j in range(len(x)):
                reward_scst_list[i,j,int(x[j])]=reward_scst[i]

    else:
        reward_scst_list=None
        base_score = 0
        explore_score = np.mean(scores[:batch_size])

    reward_list=np.zeros([batch_size,gen_result.shape[1],vocab_size])
    if alpha==0:
        return reward_list,reward_scst_list,base_score, explore_score
            
    for i in range(batch_size):
        x=res[i][0].split(' ')
        if len(x)>=scorer.N:
            for j in range(len(x)):
                for n in range(scorer.N):
                    key_list=keys_res[i][0][n]
                    if use_one_gram==0:
                        if n==0:
                            continue

                    x_with_bos=[scorer.bos]*n+x
                        
                    if use_bos==False and j<n:
                        continue
                        
                    prefix=tuple(x_with_bos[j:j+n])

                    #tar_key=np.zeros([vocab_size])
                    for k,y in enumerate(key_list):
                        if y==0:
                            continue
                        if y[:n]==prefix:
                            if x[j]==y[n]:
                                reward_list[i,j,int(y[n])]+=scores_dif_tc[i,0,n,k]
                                scores_dif_tc_[i][0][n][k]=0
                            else:
                                reward_list[i,j,int(y[n])]+=scores_dif_tc_[i,0,n,k]
                                                                              
            
            if int(x[-1])==0:
                reward_list[i,len(x)-1,0]-=np.sum(scores_dif_l[i,0])

    return reward_list, reward_scst_list, base_score, explore_score
    
def get_wc_reward_CIDEr_new(model, fc_feats, att_feats,inputs,data,gen_result,scorer,opt):
    use_scst=getattr(opt, 'use_scst', 1)
    use_bos=getattr(opt, 'use_bos', 1)
    use_one_gram=getattr(opt, 'use_one_gram', 0)
    invert=getattr(opt, 'invert', 0)
    alpha=getattr(opt, 'current_alpha', 0)
    

    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    vocab_size=inputs.size(2) 

    gen_result = gen_result.cpu().numpy()

    res = []
    for i in range(batch_size):
        res.append([array_to_str(gen_result[i])])
    if use_scst!=0:
        greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True),
                                                Variable(att_feats.data, volatile=True))

        greedy_res = greedy_res.cpu().numpy()

        for i in range(batch_size):
            res.append([array_to_str(greedy_res[i])])

    gts = []
    for i in range(len(res)):
        i_=i % batch_size // seq_per_img
        data_gt=data['gts'][i_]
        
        if i<batch_size:
            #if len(data_gt)>5:
            #    index_list=list(range(5))
            #    random.shuffle(index_list)
            #    data_gt=[data_gt[t] for t in index_list]

            gts.append([array_to_str(data_gt[j],invert) for j in range(len(data_gt))])
        else:
            gts.append(gts[i%batch_size])

    scores,scores_dif_tc,scores_dif_l,keys_res=scorer.forward(res,gts)

    if use_scst!=0:
        reward_scst = scores[:batch_size] - scores[batch_size:]
        reward_scst_list=np.zeros([batch_size,gen_result.shape[1],vocab_size])
        base_score = np.mean(scores[batch_size:])
        explore_score = np.mean(scores[:batch_size])

        for i in range(batch_size):
            x=res[i][0].split(' ')
            for j in range(len(x)):
                reward_scst_list[i,j,int(x[j])]=reward_scst[i]

    else:
        reward_scst_list=None
        base_score = 0
        explore_score = np.mean(scores[:batch_size])

    reward_list=np.zeros([batch_size,gen_result.shape[1],vocab_size,scorer.N,scorer.N])
    if alpha==0:
        return reward_list,reward_scst_list,base_score, explore_score
            
    for i in range(batch_size):
        x=res[i][0].split(' ')
        if len(x)>=scorer.N:
            for j in range(len(x)):
                for n in range(scorer.N):
                    key_list=keys_res[i][0][n]

                    x_with_bos=[scorer.bos]*n+x
                        
                    if use_bos==False and j<n:
                        continue
                        

                    #tar_key=np.zeros([vocab_size])
                    for k,y in enumerate(key_list):
                        if y==0:
                            continue
                        for m in range(n+1):
                            beta=1
                            if use_one_gram==0:
                                if m==0:
                                    continue
                            else:
                                if m==0:
                                    beta=use_one_gram
                                    
                            prefix=tuple(x_with_bos[j+n-m:j+n])
                            if y[:m]==prefix and j+n-m<len(x):
                          
                                reward_list[i,j,int(y[m]),n,m]=max(reward_list[i,j,int(y[m]),n,m],scores_dif_tc[i,0,n,k]*(opt.gamma**(n-m))*beta)

                                                                              
            
            if int(x[-1])==0:
                if np.sum(scores_dif_l[i,0])>0:
                    reward_list[i,len(x)-1,0,:,:]-=np.sum(scores_dif_l[i,0])/(scorer.N**2)
                elif len(x)>=2:
                    reward_list[i,len(x)-2,0,:,:]-=np.sum(scores_dif_l[i,0])/(scorer.N**2)
                    
    reward_list=np.sum(np.sum(reward_list,4),3)

    return reward_list, reward_scst_list, base_score, explore_score

def get_wc_reward_Bleu(model, fc_feats, att_feats,inputs,data,gen_result,scorer,opt):
    use_scst=getattr(opt, 'use_scst', 1)
    use_bos=getattr(opt, 'use_bos', 1)
    use_one_gram=getattr(opt, 'use_one_gram', 0)
    invert=getattr(opt, 'invert', 0)
    alpha=getattr(opt, 'current_alpha', 0)
    

    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    vocab_size=inputs.size(2) 

    gen_result = gen_result.cpu().numpy()

    res = []
    for i in range(batch_size):
        res.append([array_to_str(gen_result[i])])
    if use_scst!=0:
        greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True),
                                                Variable(att_feats.data, volatile=True))

        greedy_res = greedy_res.cpu().numpy()

        for i in range(batch_size):
            res.append([array_to_str(greedy_res[i])])

    gts = []
    for i in range(len(res)):
        i_=i % batch_size // seq_per_img
        data_gt=data['gts'][i_]
        
        if i<batch_size:
            #if len(data_gt)>5:
            #    index_list=list(range(5))
            #    random.shuffle(index_list)
            #    data_gt=[data_gt[t] for t in index_list]

            gts.append([array_to_str(data_gt[j],invert) for j in range(len(data_gt))])
        else:
            gts.append(gts[i%batch_size])

    scores,scores_list,g_judge,scores_dif_l,keys_res=scorer.forward(res,gts)

    if use_scst!=0:
        reward_scst = scores[:batch_size] - scores[batch_size:]
        reward_scst_list=np.zeros([batch_size,gen_result.shape[1],vocab_size])
        base_score = np.mean(scores[batch_size:])
        explore_score = np.mean(scores[:batch_size])

        for i in range(batch_size):
            x=res[i][0].split(' ')
            for j in range(len(x)):
                reward_scst_list[i,j,int(x[j])]=reward_scst[i]

    else:
        reward_scst_list=None
        base_score = 0
        explore_score = np.mean(scores[:batch_size])

    reward_list=np.zeros([batch_size,gen_result.shape[1],vocab_size])
    if alpha==0:
        return reward_list,reward_scst_list,base_score, explore_score

    reward_list_tar=np.zeros([batch_size,gen_result.shape[1],vocab_size,scorer.N],dtype=np.int)
            
    for i in range(batch_size):
        x=res[i][0].split(' ')
        if len(x)>=scorer.N:
            for j in range(len(x)):
                for n in range(scorer.N):
                    key_list=keys_res[i][0][n]
                    if use_one_gram==0:
                        if n==0:
                            continue

                    x_with_bos=[scorer.bos]*n+x
                        
                    if use_bos==False and j<n:
                        continue
                        
                    prefix=tuple(x_with_bos[j:j+n])

                    #tar_key=np.zeros([vocab_size])
                    for k,y in enumerate(key_list):
                        if y==0:
                            continue
                        if y[:n]==prefix:
                            if y[n]==x_with_bos[j+n]:
                                reward_list_tar[i,j,:,n]-=int(g_judge[i][0][n][k])
                            reward_list_tar[i,j,int(y[n]),n]+=int(g_judge[i][0][n][k])
                            
                    for k,y in enumerate(key_list):
                        if y==0:
                            continue
                        if y[:n]==prefix:
                            reward_list[i,j,int(y[n])]=scores_list[i][0][tuple(reward_list_tar[i,j,int(y[n])])]

                    #reward_list[i][j][tar_key==0]+=scores_dif_tc[i][0][n][-1]                                               
            if int(x[-1])==0:
                reward_list[i][len(x)-1][0]-=np.sum(scores_dif_l[i][0])

    return reward_list, reward_scst_list, base_score, explore_score


            

                                





            





