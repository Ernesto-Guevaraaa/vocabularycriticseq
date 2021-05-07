from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
import time
import math

import pickle

def word_seg(sent):
    return sent.split(' ')
    
def cut_eos(sent,eos='0'):
    sent_seg=word_seg(sent)
    if sent_seg[-1]==eos:
        sent_seg=sent_seg[:-1]
    return ' '.join(sent_seg)

class Bleu(nn.Module):
    def __init__(self,len_max,N=4,bos=-1,use_bos=True,mode='closest',eos='0',use_eos=1):
        super(Bleu,self).__init__()
        self.N=N
        self.bos=bos
        self.len_max=len_max
        self.use_bos=use_bos
        self.mode=mode
        self.eos=eos
        self.use_eos=use_eos

    def forward(self,res,gts,g_p_i=0):
    
        if self.use_eos==0:
            res=[[cut_eos(sent,self.eos) for sent in sents] for sents in res]
            gts=[[cut_eos(sent,self.eos) for sent in sents] for sents in gts]
            
        #r_p_i must be 1
        r_p_i=len(res[0])
        if g_p_i==0:
            g_p_i=max([len(x) for x in gts])

        #g:[batch,*_p_i,N,2*len_max] l:[batch,*_p_i] key:[batch,*_p_i,N,2*len_max]
        tc_gts,g_gts,l_gts,l_nc_gts,key_gts=self.process(gts,(g_p_i+1)*self.len_max+1,ref=None,c_p_i=g_p_i)
        tc_res,g_res,l_res,l_nc_res,key_res=self.process(res,(g_p_i+1)*self.len_max+1,ref=key_gts,c_p_i=r_p_i)

        # hook=tc_res.register_hook(lambda grad: grad)

        score=[]
        loss=0
        score_list=np.zeros([tc_res.size(0),r_p_i]+[3]*self.N)

        for i in range(r_p_i):
            g_=g_res[:,i,:,:]
            g=g_
            #g_=torch.unsqueeze(g_,dim=1)
            #g=g_.expand_as(g_gts)

            l_=l_res[:,i]
            l_=torch.unsqueeze(l_,dim=1)
            l=l_.expand_as(l_gts)
            
            if self.mode=='closest':
                l_gts[l_gts==0]=self.len_max*10
                l_gts_closest=torch.argmin(torch.abs(l-l_gts),dim=1)
                l_r=l_gts.gather(1,torch.unsqueeze(l_gts_closest,1)).squeeze()

            l_factor=torch.exp(1-torch.max(torch.ones_like(l_r)+1e-9,(l_r+1e-9)/(l_res[:,i]+1e-9)))
            
            g_gts,_=torch.max(g_gts,dim=1)
            g_factor=torch.min(g,g_gts+1e-9)
            #g_factor,_=torch.max(g_factor,dim=1)
            g_factor=torch.sum(g_factor,dim=2)
            #print(g_factor)

            und=g_factor*1
            if self.use_bos==0:
                for n in range(self.N):
                    und[:,n]=1/(l[:,0]-n+1e-9)
            else:
                for n in range(self.N):
                    und[:,n]=1/(l[:,0]+1e-9)
            
            g_factor=g_factor*und
            
            g_factor_sum=torch.sum(g_factor)
            g_factor_sum.backward()
            g_judge=tc_res.grad.data.numpy()
            g_judge=(g_judge!=0)
            tc_res.grad=None

            with torch.no_grad():
                #g_factor_prod=torch.prod(g_factor,dim=1)
                g_factor_plus1=g_factor+und
                g_factor_minus1=F.relu(g_factor-und)                   

                g_delta_list=torch.zeros([g_factor.size(0)]+[3]*self.N,dtype=torch.double).cuda()
                if self.N==4:
                    for j1 in range(3):
                        for j2 in range(3):
                            for j3 in range(3):
                                for j4 in range(3):
                                    g_factor_tmp=g_factor*1
                                    ch_list=tuple((g_factor,g_factor_plus1,g_factor_minus1))
                                    for loc,idx in enumerate((j1,j2,j3,j4)):
                                        g_factor_tmp[:,loc]=ch_list[idx][:,loc]
                                    
                                    g_factor_prod_tmp=torch.prod(g_factor_tmp,dim=1)
                                    g_delta_list[:,j1,j2,j3,j4]=g_factor_prod_tmp

            g_delta_list_=g_delta_list.view(-1,3**self.N)

            c_delta_list_=torch.unsqueeze(l_factor,1).expand_as(g_delta_list_)*(g_delta_list_**(1/self.N))
            c_delta_list=c_delta_list_-torch.unsqueeze(c_delta_list_[:,-1],1).expand_as(c_delta_list_)
            c_delta_list=c_delta_list.view_as(g_delta_list)
            score_list[:,i]=c_delta_list.cpu().data.numpy()
   
            c=c_delta_list_[:,0]
            #c=l_factor*(g_factor_prod**(1/self.N))
            
            # g_factor=torch.log(g_factor)
            # g_factor=torch.sum(g_factor,dim=1)/self.N
            # c=l_factor*torch.exp(g_factor)

            loss+=torch.sum(c)
            score.append(c.cpu().data.numpy())
        
        l_nc_res.grad=None
        loss.backward()

        score=np.array(score).transpose((1,0))
        score_dif_l=l_nc_res.grad.data.numpy()

        # hook.remove()

        return score,score_list,g_judge,score_dif_l,key_res


    def process(self,inputs,key_length,ref=None,c_p_i=0):
        gram_count=[]
        length=np.zeros([len(inputs),c_p_i])
        for i,batch in enumerate(inputs):
            gram_count.append([])
            for j in range(c_p_i):
                if j<len(batch):
                    cap=batch[j]
                    cap_=word_seg(cap)
                else:
                    cap_=[]
                    
                gram_count[i].append([])
            
                length[i][j]=len(cap_)
                if len(cap_)==0:
                    length[i][j]=self.len_max*0.5
                    
                for n in range(self.N):
                    gram_count[i][j].append({})
                    if self.use_bos==True:
                        cap_with_bos=[self.bos]*(n)+cap_
                    else:
                        cap_with_bos=cap_
                    for k in range(len(cap_)):
                        if k+n+1<=len(cap_with_bos):
                            x=tuple(cap_with_bos[k:k+n+1])
                            try:
                                gram_count[i][j][n][x]=gram_count[i][j][n][x]+1
                            except:
                                gram_count[i][j][n][x]=1

        key=[]
        if ref==None:
            for i,batch in enumerate(gram_count):
                key.append([])
                for j in range(c_p_i):
                    key[i].append([])
        
                for n in range(self.N):
                    x=set()
                    for j in range(c_p_i):
                        x=x|set(gram_count[i][j][n].keys())
                    x=list(x)
                    for _ in range(key_length-len(x)):
                        x.append(0)
                    for j in range(c_p_i):
                        key[i][j].append(x)


        else:
            for i,batch in enumerate(gram_count):
                key.append([])
                for j in range(c_p_i):
                    key[i].append([])
                    for n in range(self.N):
                        x=ref[i][0][n][:key_length-self.len_max]
                        y=gram_count[i][j][n].keys()
                        x=x+list(set(y)-set(x))

                        for _ in range(key_length-len(x)):
                            x.append(0)

                        key[i][j].append(x)
 

        tc=np.zeros([len(inputs),c_p_i,self.N,key_length])
        idf=np.ones([len(inputs),c_p_i,self.N,key_length])
        for i,batch in enumerate(inputs):
            for j in range(len(batch)):
                for n in range(self.N):
                    key_list=key[i][j][n]
                        
                    for t in range(key_length):
                        x=key_list[t]
                        try:
                            tc_=gram_count[i][j][n][x]
                            tc[i][j][n][t]=tc_
                        except:
                            pass
                        try:
                            idf_=self.df['ref_len']/max(1.,self.df['document_frequency'][x])
                            idf[i][j][n][t]=idf_
                        except:
                            pass


        tc=torch.tensor(tc,requires_grad=True)
        tc_cuda=tc.cuda()
        length_nc=torch.tensor(length,requires_grad=True)
        length=length_nc.cuda()

        return tc,tc_cuda,length,length_nc,key

class Bleu_stable(Bleu):
    def __init__(self,len_max,N=4,bos=-1,use_bos=True,mode='closest',eos='0',use_eos=1):
        super(Bleu_stable,self).__init__(len_max,N,bos,use_bos,mode,eos,use_eos)

    def forward(self,res,gts,g_p_i=0):
    
        if self.use_eos==0:
            res=[[cut_eos(sent,self.eos) for sent in sents] for sents in res]
            gts=[[cut_eos(sent,self.eos) for sent in sents] for sents in gts]
    
        #r_p_i must be 1
        r_p_i=len(res[0])
        if g_p_i==0:
            g_p_i=max([len(x) for x in gts])

        #g:[batch,*_p_i,N,2*len_max] l:[batch,*_p_i] key:[batch,*_p_i,N,2*len_max]
        tc_gts,g_gts,l_gts,l_nc_gts,key_gts=self.process(gts,(g_p_i+1)*self.len_max+1,ref=None,c_p_i=g_p_i)
        tc_res,g_res,l_res,l_nc_res,key_res=self.process(res,(g_p_i+1)*self.len_max+1,ref=key_gts,c_p_i=r_p_i)

        # hook=tc_res.register_hook(lambda grad: grad)

        score=[]
        loss=0
        score_list=np.zeros([tc_res.size(0),r_p_i]+[3]*self.N)

        for i in range(r_p_i):
            g_=g_res[:,i,:,:]
            g=g_
            #g_=torch.unsqueeze(g_,dim=1)
            #g=g_.expand_as(g_gts)

            l_=l_res[:,i]
            l_=torch.unsqueeze(l_,dim=1)
            l=l_.expand_as(l_gts)
            
            if self.mode=='closest':
                l_gts[l_gts==0]=self.len_max*10
                l_gts_closest=torch.argmin(torch.abs(l-l_gts),dim=1)
                l_r=l_gts.gather(1,torch.unsqueeze(l_gts_closest,1)).squeeze()
                
            l_factor=torch.exp(1-torch.max(torch.ones_like(l_r)+1e-9,(l_r+1e-9)/(l_res[:,i]+1e-9)))
            
            g_gts,_=torch.max(g_gts,dim=1)
            g_factor=torch.min(g,g_gts+1e-9)
            #g_factor,_=torch.max(g_factor,dim=1)
            g_factor=torch.sum(g_factor,dim=2)
            #print(g_factor)
            und=g_factor*1
            if self.use_bos==0:
                for n in range(self.N):
                    und[:,n]=1/(l[:,0]-n+1e-9)
            else:
                for n in range(self.N):
                    und[:,n]=1/(l[:,0]+1e-9)
            g_factor_with_und=g_factor*und
            
            g_factor_nonzero=(g_factor+0.008*(g_factor==0).double())*und
            
            g_factor_prod_u=torch.sum(g_factor_with_und)
            g_factor_prod_u.backward()
            g_judge=tc_res.grad.data.numpy()
            g_judge=(g_judge!=0)
            tc_res.grad=None

            with torch.no_grad():
                #g_factor_prod=torch.prod(g_factor,dim=1)
                g_factor_plus1=g_factor_nonzero+und
                g_factor_minus1=F.relu(g_factor_nonzero-und)                   

                g_delta_list=torch.zeros([g_factor.size(0)]+[3]*self.N,dtype=torch.double).cuda()
                if self.N==4:
                    for j1 in range(3):
                        for j2 in range(3):
                            for j3 in range(3):
                                for j4 in range(3):
                                    g_factor_tmp=g_factor_nonzero*1
                                    ch_list=tuple((g_factor_nonzero,g_factor_plus1,g_factor_minus1))
                                    for loc,idx in enumerate((j1,j2,j3,j4)):
                                        g_factor_tmp[:,loc]=ch_list[idx][:,loc]
                                    
                                    g_factor_prod_tmp=torch.prod(g_factor_tmp,dim=1)
                                    g_delta_list[:,j1,j2,j3,j4]=g_factor_prod_tmp

            g_delta_list_=g_delta_list.view(-1,3**self.N)

            c_delta_list_=torch.unsqueeze(l_factor,1).expand_as(g_delta_list_)*(g_delta_list_**(1/self.N))
            c_delta_list=c_delta_list_-torch.unsqueeze(c_delta_list_[:,-1],1).expand_as(c_delta_list_)
            c_delta_list=c_delta_list.view_as(g_delta_list)
            score_list[:,i]=c_delta_list.cpu().data.numpy()
   
            c=c_delta_list_[:,0]
            #c=l_factor*(g_factor_prod**(1/self.N))
            
            # g_factor=torch.log(g_factor)
            # g_factor=torch.sum(g_factor,dim=1)/self.N
            # c=l_factor*torch.exp(g_factor)

            loss+=torch.sum(c)
            score.append(c.cpu().data.numpy())
        
        l_nc_res.grad=None
        loss.backward()

        score=np.array(score).transpose((1,0))
        score_dif_l=l_nc_res.grad.data.numpy()

        # hook.remove()

        return score,score_list,g_judge,score_dif_l,key_res

if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='6' 
    scorer=Bleu_stable(20,use_bos=0)
    res=[['a plate of man sitting on a table'],['a woman is walking a a a'],['there is an apple on a table']]
    gts=[['a plate full of food sitting on a table', 'a close shot of french fries with gravy', 'a plate of food and a glass on a table', 'a plate with fries covered in gravy sitting on a table', 'a plate of fries covered in a white sauce'],['a woman is walking a dog','a beautiful woman is walking with a dog','there is a woman with a dog'],['an apple on a table','there is a table and an apple','a picture with an apple and a table']]

    # res=[['a group of elephants walking in a zoo']]
    # gts=[['several elephants are in a habitat as heads are in the foreground','a small gray elephant standing in an exhibit at a zoo', 'people are watching four elephants in a zoo', 'several elephants in zoo enclosure with onlookers watching', 'an elephant in a zoo stands in front of the crowd']]

    score,score_list,g_judge,score_dif_l,key_res=scorer.forward(res,gts)
    print(np.shape(score_list))
    print(np.shape(g_judge))
    print(np.shape(score_dif_l))
    print('score:')
    print(score)

    print('l:')
    print(score_dif_l)

    import sys 
    sys.path.append('/home/liuhan/multi-mode/coco-caption')
    from pycocoevalcap.bleu.bleu import Bleu
    scorer=Bleu()
    #res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    res_={i: res[i] for i in range(len(res))}
    gts_ = {i: gts[i] for i in range(len(gts))}
    _, scores = scorer.compute_score(gts_, res_)
    print(scores[-1])
    #print('g_dif:')
    # print(cider_dif_g)
    # print('l_dif:')
    # print(cider_dif_l)
    # scorer=cider_scorer
    # scores,scores_dif_tc,scores_dif_l,keys_res=scorer.forward(res,gts)

    # reward_list=np.zeros([2,20])
    # for i in range(len(res)):
    #     x=res[i][0].split(' ')

    #     for j in range(len(x)):
    #         reward=0
    #         for n in range(scorer.N):
    #             key_list=keys_res[i][0][n]
    #             for m in range(n+1):
    #                 if m>0:
    #                     if j>=m:
    #                         prefix=tuple(x[j-m:j])
    #                         for k,y in enumerate(key_list):
    #                             if y==0:
    #                                 break
    #                             if y[:m]==prefix:
    #                                 if x[j]==y[m]:
    #                                     reward+=scores_dif_tc[i][0][n][k]/(n+1)
    #                                 else:
    #                                     reward-=m*scores_dif_tc[i][0][n][k]/(n+1)
    #                 else:
    #                     for k,y in enumerate(key_list):
    #                         if y==0:
    #                             break
    #                         if x[j]==y[m]:
    #                             reward+=scores_dif_tc[i][0][n][k]/(n+1)
    #         reward_list[i][j]=reward

    #     if len(x)<20:
    #         j=len(x)
    #         reward=0
    #         for n in range(scorer.N):
    #             key_list=keys_res[i][0][n]
    #             for m in range(n+1):
    #                 if m>0:
    #                     if j>=m:
    #                         prefix=tuple(x[j-m:j])
    #                         for k,y in enumerate(key_list):
    #                             if y==0:
    #                                 break
    #                             if y[:m]==prefix:
    #                                 reward-=m*scores_dif_tc[i][0][n][k]/(n+1)
    #         reward-=np.sum(scores_dif_l[i][0])
    #         reward_list[i][j]=reward
    
    # print(reward_list)

        
        
        

                        

                    


        


        
            
            


                
                





        
