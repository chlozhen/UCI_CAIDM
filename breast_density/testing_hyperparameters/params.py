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
          }
    #CURRENT TEST
    nfold = 1
    lr = [.001,5e-4,1e-4,5e-5,1e-5]
#     #TODO
#     alpha = [.5, 1.5, 2.5]
#     batch_size = [8, 12]
#     epochs, steps_per_epoch

#     #TESTED
#     dropout = [.005, .01, .1, .15, .2, .3, .4] *.1
#     reg = ['l1', 'l2'] *l2
#     reg_val = [1e-2, 1e-3, 1e-4] *1e-4
    
#     # --- Create hyper-lreg-
#     # SET: LR = .001, fold = 1, batch = 12, alpha = 1, dropout = 0
#     # TESTING: regularizer
#     exp = 0
#     for fold in range(nfold):
#         for v in range(len(reg_val)):
#             for r in reg:
#                 df['output_dir'].append('{}/exp-{}'.format(log_path, exp))
#                 df['LR'].append(.001)
#                 df['fold'].append(fold)
#                 df['batch_size'].append(12)
#                 df['alpha'].append(1)
#                 df['dropout'].append(0)
#                 df['reg'].append(r)
#                 df['reg_val'].append(reg_val[v])
#                 exp += 1
        
#     # --- Create hyper-drop-
#     # SET: LR = .001, fold = 1, batch = 12, alpha = 1, reg = l2, reg_val= .0001
#     # TESTING: dropout
#     exp = 0
#     for d in range(len(dropout)):
#         df['output_dir'].append('{}/exp-{}'.format(log_path, exp))
#         df['LR'].append(.001)
#         df['fold'].append(0)
#         df['batch_size'].append(12)
#         df['alpha'].append(1)
#         df['dropout'].append(dropout[d])
#         df['reg'].append('l2')
#         df['reg_val'].append(.0001)
#         exp += 1

    # --- Create hyper-lr-
    # SET: LR = .001, fold = 1, batch = 12, alpha = 1, reg = l2, reg_val= .0001, dropout = .1
    # TESTING: lr
    exp = 0
    for l in range(len(lr)):
        df['output_dir'].append('{}/exp-{}'.format(log_path, exp))
        df['LR'].append(lr[l])
        df['fold'].append(0)
        df['batch_size'].append(12)
        df['alpha'].append(1)
        df['dropout'].append(.1)
        df['reg'].append('l2')
        df['reg_val'].append(.0001)
        exp += 1
    
    # --- Save *.csv file
    df = pd.DataFrame(df)
    df.to_csv(fname, index=False)
    print('Created {} successfully'.format(fname))
    
# --- NOTE: Creates hyper csv and log directory
if __name__ == '__main__':
    fname = 'hyper_lrt_sched' #CHANGE EXPNAME
    outpath = r'/home/chloez/breast_density/model_perm/logs'
    create_hyper_csv(fname, outpath)
    

    #notes
    #l2(.0001) - not significant
    #dropout   - not significant
    