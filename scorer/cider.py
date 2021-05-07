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

class CIDEr_D(nn.Module):
    def __init__(self,df,len_max,N=4,sigma=6,bos=-1,use_bos=True,eos='0',use_eos=True):
        super(CIDEr_D,self).__init__()
        self.df=df
        self.N=N
        self.bos=bos
        self.len_max=len_max
        self.sigma=sigma
        self.use_bos=use_bos
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

        #g:[batch,*_p_i,N,2*len_max] l:[batch,*_p_i,N] key:[batch,*_p_i,N,2*len_max]
        tc_gts,g_gts,l_gts,l_nc_gts,key_gts=self.process(gts,(g_p_i+1)*self.len_max+1,ref=None,c_p_i=g_p_i)
        tc_res,g_res,l_res,l_nc_res,key_res=self.process(res,(g_p_i+1)*self.len_max+1,ref=key_gts,c_p_i=r_p_i)

        # hook=tc_res.register_hook(lambda grad: grad)
        gts_count=np.array([len(gt) for gt in gts],dtype='float64')
        gts_count=torch.from_numpy(gts_count).cuda()

        score=[]
        loss=0
        for i in range(r_p_i):
            g_=g_res[:,i,:,:]
            g_=torch.unsqueeze(g_,dim=1)
            g=g_.expand_as(g_gts)

            l_=l_res[:,i,:]
            l_=torch.unsqueeze(l_,dim=1)
            l=l_.expand_as(l_gts)

            l_factor=torch.exp(-1*((l-l_gts)*(l-l_gts))/(2*self.sigma**2))
            inner_product=torch.sum(torch.min(g,g_gts+1e-9)*g_gts,dim=3)
            with torch.no_grad():
                norm=torch.sqrt((torch.sum(g*g,dim=3)+1e-9)*(torch.sum(g_gts*g_gts,dim=3)+1e-9))
                
            g_factor=inner_product/norm
            
            c=l_factor*g_factor
            gts_count_=torch.unsqueeze(torch.unsqueeze(gts_count,1),2).expand_as(c)
            c=c/gts_count_
            c=10*torch.sum(c,dim=1)

            c=torch.sum(c,dim=1)/self.N
            loss+=torch.sum(c)
            score.append(c.cpu().data.numpy())
        loss.backward()

        score=np.array(score).transpose((1,0))
        score_dif_tc=tc_res.grad.data.numpy()
        score_dif_l=l_nc_res.grad.data.numpy()

        # hook.remove()

        return score,score_dif_tc,score_dif_l,key_res

    # def process(self,inputs,key_length,ref=None):
    #     gram_count=[]
    #     length=np.zeros([len(inputs),len(inputs[0])])
    #     for i,batch in enumerate(inputs):
    #         gram_count.append([])
    #         for j,cap in enumerate(batch):
    #             gram_count[i].append([])
    #             cap_=word_seg(cap)
    #             length[i][j]=len(cap_)
    #             for n in range(self.N):
    #                 gram_count[i][j].append({})
    #                 for k in range(len(cap_)-n):
    #                     x=tuple(cap_[k:k+n+1])
    #                     gram_count[i][j][n][x]=gram_count[i][j][n].get(x,0)+1


    #     key=[]
    #     if ref==None:
    #         for i,batch in enumerate(gram_count):
    #             key.append([])
    #             for j,cap in enumerate(batch):
    #                 key[i].append([])
        
    #             for n in range(self.N):
    #                 x=set()
    #                 for j in range(len(batch)):
    #                     x=x|set(gram_count[i][j][n].keys())
    #                 x=list(x)
    #                 for _ in range(key_length-len(x)):
    #                     x.append(0)
    #                 for j in range(len(batch)):
    #                     key[i][j].append(x)


    #     else:
    #         for i,batch in enumerate(gram_count):
    #             key.append([])
    #             for j,cap in enumerate(batch):
    #                 key[i].append([])
    #                 for n in range(self.N):
    #                     x=ref[i][0][n][:key_length-self.len_max]
    #                     y=gram_count[i][j][n].keys()
    #                     x=x+list(set(y)-set(x))

    #                     for _ in range(key_length-len(x)):
    #                         x.append(0)

    #                     key[i][j].append(x)
 

    #     tc=np.zeros([len(inputs),len(inputs[0]),self.N,key_length])
    #     idf=np.zeros([len(inputs),len(inputs[0]),self.N,key_length])
    #     for i,batch in enumerate(gram_count):
    #         for j,cap in enumerate(batch):
    #             for n in range(self.N):
    #                 key_list=key[i][j][n]
                        
    #                 for t in range(key_length):
    #                     x=key_list[t]
    #                     tc_=gram_count[i][j][n].get(x,0)
    #                     idf_=self.df['ref_len']/max(1.,self.df['document_frequency'].get(x,1))

    #                     tc[i][j][n][t]=tc_
    #                     idf[i][j][n][t]=idf_

    #     tc=torch.tensor(tc,requires_grad=True)
    #     tc_cuda=tc.cuda()
    #     idf=torch.tensor(idf,requires_grad=False).cuda()
    #     length_nc=torch.tensor(length,requires_grad=True)
    #     length=length_nc.cuda()
    #     length=torch.unsqueeze(length,2)
    #     length=torch.cat([length]*self.N,2)
    #     # g=tc_cuda*torch.log(idf)/torch.unsqueeze(torch.unsqueeze(length_ngram,2),3).expand_as(idf)
    #     g=tc_cuda*torch.log(idf)

    #     return tc,g,length,length_nc,key

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
        idf=torch.tensor(idf,requires_grad=False).cuda()
        length_nc=torch.tensor(length,requires_grad=True)
        length=length_nc.cuda()
        length=torch.unsqueeze(length,2)
        length=torch.cat([length]*self.N,2)
        # g=tc_cuda*torch.log(idf)/torch.unsqueeze(torch.unsqueeze(length_ngram,2),3).expand_as(idf)
        g=tc_cuda*torch.log(idf)

        return tc,g,length,length_nc,key

if __name__=='__main__':
    df_path='/home/liuhan/dataset/coco/coco-all-words.p'
    pkl_file = pickle.load(open(df_path))
    cider_scorer=CIDEr_D(pkl_file,20,use_bos=0)
    res=[['a woman is walking a dog'],['there is an apple on the table']]
    gts=[['a woman is walking a dog','a beautiful woman is walking with a dog','there is a woman with a dog'],['an apple on a table','there is a table and an apple','a picture with an apple and a table']]

    # res=[['a group of elephants walking in a zoo']]
    # gts=[['several elephants are in a habitat as heads are in the foreground','a small gray elephant standing in an exhibit at a zoo', 'people are watching four elephants in a zoo', 'several elephants in zoo enclosure with onlookers watching', 'an elephant in a zoo stands in front of the crowd']]

    cider,cider_dif_g,cider_dif_l,key_res=cider_scorer.forward(res,gts)
    print(np.shape(cider_dif_g))
    print(np.shape(cider_dif_l))
    print('cider:')
    print(cider)
    dif_key=[]
    for i,x in enumerate(key_res):
        dif_key.append([])
        for j,y in enumerate(x):
            dif_key[i].append([])
            for n in range(len(y)):
                dif_key[i][j].append({})
                for m,key in enumerate(y[n]):
                    if key!=0:
                        dif_key[i][j][n][key]=cider_dif_g[i][j][n][m]
    print(dif_key[0][0][0])
    #print(dif_key[0][0][1])
    # print(dif_key[0][0][2])
    print(cider_dif_g[0][0][1][-1])
    print(cider_dif_l)

    import sys 
    sys.path.append('..')
    from ciderD.pyciderevalcap.ciderD.ciderD import CiderD
    CiderD_scorer=CiderD(df='/home/liuhan/dataset/coco/coco-all-words')
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    gts_ = {i: gts[i] for i in range(len(gts))}
    _, scores = CiderD_scorer.compute_score(gts_, res_)
    print(scores)
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

        
        
        

                        

                    


        


        
            
            


                
                





        
