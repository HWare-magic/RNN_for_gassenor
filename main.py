import sys
import os
import argparse
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import lib.Metrics
import lib.Utils
## TODO
from model.AGCRN import *


N_NODE = 207
CHANNEL = 1
LEARN = 0.001
PATIENCE = 10
OPTIMIZER = 'Adam'
# OPTIMIZER = 'RMSprop'
# LOSS = 'MSE'
LOSS = 'MAE'
TRAINRATIO = 0.8 # TRAIN + VAL
TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1

################# python input parameters #######################
parser = argparse.ArgumentParser()
## TODO
parser.add_argument('-model',type=str,default='AGCRN',help='choose which model to train and test')
parser.add_argument('-version',type=int,default=0,help='train version')
parser.add_argument('-instep',type=int,default=12,help='input step')
parser.add_argument('-outstep',type=int,default=12,help='predict step')
parser.add_argument('-batch',type=int,default=128,help='batch size')
parser.add_argument('-epoch',type=int,default=1,help='training epochs')
parser.add_argument('-mode',type=str,default='train',help='train or eval')
parser.add_argument('-data',type=str,default='metrla',help='METR-LA or PEMS-BAY or PEMSD7M')
parser.add_argument('cuda',type=int,default=3,help='cuda device number')
args = parser.parse_args() #python
# args = parser.parse_args(args=[])    #jupyter notebook
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu") 
#### DATA SELECTION
if args.data=='metrla':
    FLOWPATH = './data/METRLA/metr-la.h5'
    ADJPATH = './data/METRLA/W_metrla.csv'
    DATANAME = 'METR-LA'
#### Param setting    
MODELNAME = args.model
BATCHSIZE = args.batch
EPOCH = args.epoch
TIMESTEP_IN = args.instep
TIMESTEP_OUT = args.outstep
#################################################################
def getTimestamp(data):
    # data is a pandas dataframe with timestamp ID.
    data_feature = data.values.reshape(data.shape[0],data.shape[1],1)
    feature_list = [data_feature]
    num_samples, num_nodes = data.shape
    time_ind = (data.index.values - data.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [num_nodes,1]).transpose((1, 0))
    return time_in_day

def getXSYSTIME(data, data_time, mode):
    # When CHANNENL = 2, use this function to get data plus time as two channels.
    # data: numpy, data_time: numpy from getTimestamp 
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS, XS_TIME = [], [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            t = data_time[i:i+TIMESTEP_IN, :]
            XS.append(x), YS.append(y), XS_TIME.append(t)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            t = data_time[i:i+TIMESTEP_IN, :]
            XS.append(x), YS.append(y), XS_TIME.append(t)
    XS, YS, XS_TIME = np.array(XS), np.array(YS), np.array(XS_TIME)
    XS = np.concatenate([np.expand_dims(XS, axis=-1), np.expand_dims(XS_TIME, axis=-1)], axis=-1)
    XS, YS = XS.transpose(0, 3, 2, 1), YS[:, :, :, np.newaxis]
    print('XS shape is: ',XS.shape)
    print('YS shape is: ',YS.shape)
    return XS, YS

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    return XS, YS

def getModel(name):
    model = AGCRN(num_nodes=N_NODE, input_dim=CHANNEL, output_dim=CHANNEL, horizon=TIMESTEP_OUT).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name)
    summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
    print('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    
    min_val_loss = np.inf
    wait = 0

    print('LOSS is :',LOSS)
    if LOSS == "MaskMAE":
        criterion = lib.Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    
    for epoch in range(EPOCH): # EPOCH
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())
        
def testModel(name, mode, XS, YS):
    if LOSS == "MaskMAE":
        criterion = lib.Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH+ '/' + name + '.pt'))
    
    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    print("3, 6, 12 pred steps evaluation: ")
    f.write("3, 6, 12 pred steps evaluation: ")
    for i in [2,5,11]:
        MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))        
    f.close()
    print('Model Testing Ended ...', time.ctime())

def multi_version_test(name, mode, XS, YS, versions):
    if LOSS == "MaskMAE":
        criterion = lib.Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    print('Model Testing Started ...', time.ctime())
    print('INPUT_STEP, PRED_STEP', TIMESTEP_IN, TIMESTEP_OUT)

    
    mse_all, rmse_all, mae_all, mape_all  = np.zeros((len(versions),TIMESTEP_OUT)),np.zeros((len(versions),TIMESTEP_OUT)),np.zeros((len(versions),TIMESTEP_OUT)),np.zeros((len(versions),TIMESTEP_OUT))
    f = open(PATH + '/' + name + '_multi_version_prediction_scores.txt', 'a')
    for v_ in versions:  
        XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
        test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
        test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
        model = getModel(name)        
        
        multi_test_PATH = '../save/' + DATANAME + '_' + name + '_' + str(v_) + '/' + name + '.pt'
        if os.path.isfile(multi_test_PATH):       
            model.load_state_dict(torch.load(multi_test_PATH))
            print('file path is : ',multi_test_PATH)
        else:
            print("file not exist")
            break       
        YS_pred = predictModel(model, test_iter)
        YS_truth = YS
        YS_truth, YS_pred = scaler.inverse_transform(np.squeeze(YS_truth)), scaler.inverse_transform(np.squeeze(YS_pred))
        MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS_truth, YS_pred)
        print('*' * 40)
        print(f'Version: {v_} Start Testing :')
        print("all pred steps in version %d, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (v_, name, mode, MSE, RMSE, MAE, MAPE))  
        for i in range(TIMESTEP_OUT):
            MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS_truth[:, i, :], YS_pred[:, i, :])
            mse_all[v_,i], rmse_all[v_,i], mae_all[v_,i], mape_all[v_,i] = MSE, RMSE, MAE, MAPE
            print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
            f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE)) 
        print("3, 6, 12 pred steps evaluation: ")
        for i in [2,5,11]:
            MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS_truth[:, i, :], YS_pred[:, i, :])
            print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
            f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        print('*'*40)
        print('--- version ',v_,' evaluation end ---')
        print('')
    mse = np.array(mse_all).mean(axis=0)
    rmse = np.array(rmse_all).mean(axis=0)
    mae = np.array(mae_all).mean(axis=0)
    mape = np.array(mape_all).mean(axis=0)
    print('*'*40)
    print('*'*40)
    print('*'*40)
    print('Results in Test Dataset in Each Horizon with All Version Average:')    
    for i in range(TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = mse[i], rmse[i], mae[i], mape[i]
        print("%d Horizon, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d Horizon, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE)) 
    print("Results in Test Dataset in 3, 6, 12 Horizon with All Version:")
    f.write("Results in Test Dataset in 3, 6, 12 Horizon with All Version:")
    for i in [2,5,11]:
        MSE, RMSE, MAE, MAPE = mse[i], rmse[i], mae[i], mape[i]
        print("%d Horizon, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d Horizon, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))    
    f.close()
    print('Model Multi Version Testing Ended ...', time.ctime())        
    
################# Parameter Setting #######################
KEYWORD = DATANAME + '_AGCRN_in' + str(args.instep) + '_out' + str(args.outstep) +'_version_' + str(args.version)
PATH = './save/' + KEYWORD
# torch.manual_seed(100)
# torch.cuda.manual_seed(100)
# np.random.seed(100)
# torch.backends.cudnn.deterministic = True
import os
cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
###########################################################
# data preparation
data = pd.read_hdf(FLOWPATH) # [samples,nodes]
timestamp = data.index
timestamp = np.tile(timestamp, [data.shape[0],1])  # [samples,nodes]
data = data.values
scaler = StandardScaler()
data = scaler.fit_transform(data)
print('data.shape', data.shape)
###########################################################
def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('./model/AGCRN.py', PATH)
#     shutil.copy2('Param.py', PATH)
    
    if args.mode == 'train':    
        print(KEYWORD, 'training started', time.ctime())
        trainXS, trainYS = getXSYS(data, 'TRAIN')
        print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
        trainModel(MODELNAME, 'train', trainXS, trainYS)

        print(KEYWORD, 'testing started', time.ctime())
        testXS, testYS = getXSYS(data, 'TEST')
        print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
        testModel(MODELNAME, 'test', testXS, testYS)
    if args.mode == 'eval':
        print(KEYWORD, 'multi version testing started', time.ctime())
        testXS, testYS = getXSYS(data, 'TEST')
        print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)        
        multi_version_test(MODELNAME, args.mode, testXS,testYS, versions=np.arange(0,3))

    
if __name__ == '__main__':
    main()

