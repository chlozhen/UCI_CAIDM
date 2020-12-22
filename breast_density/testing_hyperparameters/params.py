import os, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import losses, optimizers
from tensorflow.keras import Input, Model, layers
from jarvis.train import datasets, params
from jarvis.train.client import Client


def create_hyper_csv(fname='hyper', outpath='/home/breast_density/', overwrite=False):

    log_path = os.path.join(outpath, 'logs-{}'.format(fname))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    fname = './hyparams/{}.csv'.format(fname)
    
    if os.path.exists(fname) and not overwrite:
        print("OVERWRITING WARNING: Please rename the filename")
        return
    
    df = {'output_dir': [],
          'LR': [],
          'fold': [], 
          'batch_size': [], 
          'alpha': [], 
          'dropout': [],
          'reg': [],
          'reg_val': [],
          'steps_per_epoch':[]
          }
    
    #CURRENT TEST
#     nfold = 1    
#     #TODO
    alpha = [1,1,1]

#     #TESTED
#     steps_per_epoch = [100,200,300,400,500]
#     dropout = [.005, .01, .1, .15, .2, .3, .4] *.1
#     reg = ['l1', 'l2'] *l2
#     reg_val = [5e-1, 1e-1, 5e-2, 1e-2, 1e-3, 1e-4] #*1e-2
#     lr = [.001,5e-4,1e-4,5e-5,1e-5]
#     batch_size = [8, 12, 16]

    
    exp = 0
    for i in range(len(alpha)):
        df['output_dir'].append('{}/exp-{}'.format(log_path, exp))
        df['LR'].append(.0005)
        df['fold'].append(0)
        df['batch_size'].append(12)
        df['alpha'].append(alpha[i])
        df['dropout'].append(0)
        df['reg'].append('l2')
        df['reg_val'].append(1e-2)
        df['steps_per_epoch'].append(100)
        exp += 1
    
    # --- Save *.csv file
    df = pd.DataFrame(df)
    df.to_csv(fname, index=False)
    print('Created {} successfully'.format(fname))
    
# --- NOTE: Creates hyper csv and log directory
if __name__ == '__main__':
    fname = 'TR3_alpha' #CHANGE EXPNAME
    outpath = r'/home/chloez/breast_density/model_perm/logs'
    create_hyper_csv(fname, outpath, True)
  
    

# NOTES
    
    # --- Create hyper-lreg-
    # SET: LR = .001, fold = 1, batch = 12, alpha = 1, dropout = 0
    # TESTING: regularizer
        
    # --- Create hyper-drop-
    # SET: LR = .001, fold = 1, batch = 12, alpha = 1, reg = l2, reg_val= .0001
    # TESTING: dropout

    # --- Create hyper-lrt-
    # SET: fold = 1, batch = 12, alpha = 1, reg = l2, reg_val= .0001, dropout = .1
    # TESTING: lr... inconclusive - just start at .001 again for now
    
    # --- Create hyper-lr-sched
    # SET: fold = 1, batch = 12, alpha = 1, reg = l2, reg_val= .0001, dropout = .1
    # TESTING: lr... still pretty bad. 
    
    # --- Create hyper-lr-sched
    # SET: LR = [.001, .0005], fold = 1, batch = 12, alpha = 1, reg = l2, reg_val= .0001, dropout = .1
    # TESTING: steps_per_epoch
    
    # ---TR2: Testing Scaling and LR Scheduler
    # Make sure to adjust results to reflect scaling - scale by 25% --> divide loss by 25%
    
    # ---TR3: removing Activation function, retesting L2 function
    # 1. YAY! got less than .04 for mae --> .39
    # 2. Remove Early Stopping: Train for 40,000 steps
    
