from .bleu import *
from .cider import *

def scorer_setup(opt):
    if opt.tar_metric == 'CIDEr':
        import pickle
        df_path=opt.cached_tokens+'.p'
        df = pickle.load(open(df_path))
        scorer=CIDEr_D(df,opt.seq_length,use_bos=opt.use_bos,use_eos=opt.use_eos)
    elif opt.tar_metric=='Bleu':
        scorer=Bleu_stable(opt.seq_length,use_bos=opt.use_bos,use_eos=opt.use_eos)
    return scorer







        
        
        

                        

                    


        


        
            
            


                
                





        
