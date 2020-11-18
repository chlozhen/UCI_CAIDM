import glob, os
import numpy as np
import copy
from scipy import ndimage
import pandas as pd

from tensorflow.keras import Input, Model, layers, models, losses, optimizers, regularizers
from tensorflow import math

from jarvis.train.client import Client
from jarvis.train import datasets, params

from jarvis.tools import show
from pprint import pprint
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, EarlyStopping #, ReduceLROnPlateau

from jarvis.utils.general import gpus, tools as jtools, overload


# =======================================================
# FUNCTIONS
# =======================================================
# @overload(Client)
# def preprocess(self, arrays, **kwargs):
#     arrays['ys']['lbl'] *= 2.5
#     arrays['ys']['lbl'] = arrays['ys']['lbl'].clip(max=1.0)
#     return arrays

def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

def lr_scheduler(epoch, lr):
    return lr*.99
    
def write_results(result, path, p):
    os.makedirs(path, exist_ok=True)
    filename = 'exp-{}.csv'.format(ROW_NUM)
    outpath = os.path.join(path, filename)
    print("history len:", len(result.history['lr']))
    df = {
        'epoch': result.epoch,
        'learning_rate': result.history['lr'],
        'mae loss': result.history['loss'],
        'mse loss': result.history['mean_squared_error'],
        'mae val_loss': result.history['val_loss'],
        'mse val_loss': result.history['val_mean_squared_error'],
        'logs': p['output_dir']
    }    
    # --- Save *.csv file
    df = pd.DataFrame(df)
    df.to_csv(outpath, index=False)
    return
    
def prepare_client(path, p):
    
    paths = jtools.get_paths(path)
    client = Client('{}/data/ymls/client.yml'.format(paths['code']), configs={'batch': {'size': p['batch_size'],
                                                                                        'fold': p['fold']}})
    
    return client

def prepare_callback(path, p):
    
    # --- create weights folder
    os.makedirs(path, exist_ok=True)
    path = os.path.normpath(path) + '/exp-' + str(ROW_NUM) + '/'
    os.makedirs(path, exist_ok=True)
    path = path + 'epoch-{epoch:02d}-{mean_squared_error:.2f}.hdf5'
    
    checkpoint_callback = ModelCheckpoint(path, monitor='mean_squared_error', verbose=1, \
                             save_best_only=False, \
                             mode='auto', save_frequency=1)
    
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='auto')
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)
    reduce_lr = LearningRateScheduler(lr_scheduler)
    return [checkpoint_callback, early_stopping, reduce_lr]

def prepare_model(inputs, p):
    
    # --- load alpha
    a = p['alpha']
    rate = p['dropout']
    reg_type = p['reg'] #l1 or l2
    reg_val = p['reg_val']
    if reg_type == 'l1':
        reg = regularizers.l1(reg_val)
    elif reg_type == 'l2':
        reg = regularizers.l2(reg_val)
    else:
        reg = None

    kwargs = {
    'kernel_size': (1, 3, 3),
    'padding': 'same',
    'kernel_regularizer': reg 
    }
  #  'kernel_initializer': 'he_normal'}

    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.LeakyReLU()(x)
    drop = lambda x, rate : layers.Dropout(rate)(x)
    conv1 = lambda filters, x : relu(drop(norm(conv(x, filters, strides=1)), rate))
    conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=(1, 2, 2))))

    # T4
    l1 = conv2(16*a, conv1(16*a, conv1(16*a, inputs['dat'])))
    l2 = conv2(36*a, conv1(36*a, conv1(36*a, l1)))
    l3 = conv2(48*a, conv1(48*a, conv1(48*a, l2)))
    l4 = conv2(64*a, conv1(64*a, conv1(64*a, l3)))
    l5 = conv2(80*a, conv1(80*a, conv1(80*a, l4)))
    l6 = conv2(112*a, conv1(112*a, conv1(112*a, l5)))
    l7 = conv2(128*a, conv1(128*a, conv1(128*a, l6)))
    
    f0 = layers.Reshape((1, 1, 1, 2 * 2 * 128*a))(l7)
    logits = {}
    logits['lbl'] = layers.Conv3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid', name='lbl')(f0)
    
    model = Model(inputs=inputs, outputs=logits)
    opt = optimizers.Adam(learning_rate=p['LR'])
    lr_metric = get_lr_metric(optimizers.Adam(learning_rate=p['LR']))
    
    # --- compile model
    model.compile(
        optimizer=opt,
        loss={'lbl': losses.MeanAbsoluteError()}, 
        metrics=[losses.MeanSquaredError(), lr_metric],
        experimental_run_tf_function=False)

    return model

# --- NOTE: Creates weights folder and results csv
if __name__ == '__main__':
    # =======================================================
    # GLOBALS
    # =======================================================
    TEST = False
    hyparams = 'TR2_noScale_batch_111520_1239' #CHANGE EXPNAME
    CLIENT_PROJECT_NAME = 'xr/breast-fgt'
    CSV_PATH = '/home/chloez/breast_density/model_perm/hyparams/{}.csv'.format(hyparams)
    WEIGHTS_PATH = '/home/chloez/breast_density/model_perm/weights/weights-{}'.format(hyparams)
    RESULTS_PATH = '/home/chloez/breast_density/model_perm/results/results-{}'.format(hyparams)
    
    # --- Create short test to check for errors!
    if TEST:
        ROW_NUM = STEPS_PER_EPOCH = VALIDATION_STEPS = nepoch = 5
        p = {'output_dir': 'test', 'LR': .001,'fold': 0, 'batch_size': 12, 'alpha': 1, 'dropout': 0,'reg': 'l1', 'reg_val': 1e-2}
    else:
        # =======================================================
        # MODEL VARIABLES
        # =======================================================
        ROW_NUM = os.environ['JARVIS_PARAMS_ROW']
        p = params.load(CSV_PATH, row=ROW_NUM)
        STEPS_PER_EPOCH = p['steps_per_epoch']
        VALIDATION_STEPS = p['steps_per_epoch']
        nepoch = 400
        
    # =======================================================
    # PREPARATION 
    # =======================================================

    # --- Autoselect GPU
    gpus.autoselect()
    
    # --- prepare client
    client = prepare_client(CLIENT_PROJECT_NAME, p)

    # --- prepare checkpoint
    callback_list = prepare_callback(WEIGHTS_PATH, p)

    # --- create tensor object input files
    inputs = client.get_inputs(Input)

    # --- create generator
    gen_train, gen_valid = client.create_generators(batch_size=p['batch_size'])

    # --- prepare model
    model = prepare_model(inputs, p)

    # --- Train
    result = model.fit(
        x=gen_train, 
        epochs=nepoch,
        steps_per_epoch=STEPS_PER_EPOCH, 
        validation_data=gen_valid,
        validation_steps=VALIDATION_STEPS,
        callbacks=callback_list
    )
    
    write_results(result, RESULTS_PATH, p)
    
    print("\n.\nHyperparameters: {}\n.\n.Finished!".format(p))