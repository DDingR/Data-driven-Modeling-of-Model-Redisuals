import torch
from torch import nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import os
from time import time, localtime, strftime

from utils.others import *
from utils.reporter import *

raw_csv_dir = "0507_0621PM"

# START ========================================
# CONSTANTS ====================================
EPISODE = int(1e5)
EPOCH = int(50)
BATCH_SIZE = 256
VALIDATION_SIZE = 16
SAVE_START = int(1)
SAVE_PER_EPISODE = int(1e3)
PLOT_PER_EPISODE = int(10)
PLOT_FROM = 1e1
TEST_TRAIN_DATA_RATE = 0.1
# END ==========================================
# CONSTANTS ====================================

Ca = 756.349/(0.6*np.pi/180)
lf = 1.240
lr = 1.510
w = 0.8
m = 1644.80
Iz = 2488.892

def F(sample):
    x_dot = sample[:,0]
    y_dot = sample[:,1]    
    yaw_dot = sample[:,2]
    delta = sample[:,3]
    Frl = sample[:,4]
    Frr = sample[:,5]
    
    Fxf = 0
    Fyf = 2 * Ca * (delta - ((y_dot+lf*yaw_dot)/ x_dot))
    Fyr = 2 * Ca * (      - ((y_dot-lr*yaw_dot)/ x_dot))

    del_Fxf = 0
    del_Fxr = Frr - Frl

    x_ddot = ((Fxf * np.cos(delta) - Fyf * np.sin(delta)) + Frl+Frr) * 1/m + yaw_dot*y_dot
    y_ddot = ((Fxf * np.sin(delta) + Fyf * np.cos(delta)) + Fyr) * 1/m - yaw_dot*x_dot
    psi_ddot = ((lf * (Fxf * np.sin(delta) + Fyf * np.cos(delta)) - lr * Fyr) + w * (del_Fxf + del_Fxr)) / Iz

    ddots = np.array([x_ddot, y_ddot, psi_ddot])
    ddots = ddots.T
    return ddots 

# def my_loss(input, target):
#     device = "cuda"
#     weight = torch.tensor([1.6927, 1., 36.7849], device=device)
#     return torch.mean(weight * (input-target)**2)

def main():  
    # reporter config
    # cur_dir = os.getcwd()
    cur_time = strftime("%m%d_%I%M%p", localtime(time()))
    log_name = cur_time + ".log"
    board_name = "runs/" + cur_time
    reporter = reporter_loader("info", log_name)
    boardWriter = SummaryWriter(board_name)

    # dataset load
    raw_csv = "processed_csv_data/" + raw_csv_dir +".csv"
    dataset = csv2dataset(raw_csv)
    [sample_num, _] = dataset.shape
    reporter.info(f"Loaded csv data is {raw_csv_dir}")

    test_dataset = dataset[0:int(sample_num*TEST_TRAIN_DATA_RATE), 2:]
    train_dataset = dataset[int(sample_num*TEST_TRAIN_DATA_RATE):, 2:]
    [_, var_num] = train_dataset.shape
    var_num = var_num - 3

    # Trainer Setting
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelx = NN(var_num).to(device)
    modely = NN(var_num).to(device)
    modelr = NN(var_num).to(device)
    # reporter.info(summary(model, (1,var_num)))
    loss_fn = nn.MSELoss()
    # loss_fn = my_loss
    reporter.info(f"Device({device}) is working for train")

    optimizerx = torch.optim.Adam(modelx.parameters(), lr=1e-3)
    optimizery = torch.optim.Adam(modely.parameters(), lr=1e-3)
    optimizerr = torch.optim.Adam(modelr.parameters(), lr=1e-3)

    loss_list = []

    # result path
    onnx_path_x = "./savemodel/X"  + cur_time + "/"
    onnx_path_y = "./savemodel/Y"  + cur_time + "/"
    onnx_path_r = "./savemodel/R"  + cur_time + "/"

    fig_path = "./fig/" + cur_time + "loss.png"

    try:
        os.mkdir(onnx_path_x)
        os.mkdir(onnx_path_y)
        os.mkdir(onnx_path_r)

    except:
        os.rmdir(onnx_path_x)
        os.mkdir(onnx_path_x)
        os.rmdir(onnx_path_y)
        os.mkdir(onnx_path_y)
        os.rmdir(onnx_path_r)
        os.mkdir(onnx_path_r)

    try: 
        reporter.info(f"Train Started")
        for episode in range(EPISODE):
            with torch.no_grad():
                random_pick = np.random.choice(len(test_dataset), VALIDATION_SIZE)
                sample = test_dataset[random_pick,:]
                target = sample[:,0:3]
                input_data = sample[:,3:]

                analystic_target = F(input_data)
                target = target-analystic_target

                X_list = np2tensor(input_data, device)    
                target = np2tensor(target, device)

                # X_list = torch.nn.functional.normalize(X_list, dim=2)

                predx = modelx(X_list)
                predy = modely(X_list)
                predr = modelr(X_list)

                # loss = torch.sqrt(loss_fn(pred, target)).item()
                lossx = loss_fn(predx[:,:,0],target[:,:,0]).item()
                lossy = loss_fn(predy[:,:,0],target[:,:,1]).item()
                lossr = loss_fn(predr[:,:,0],target[:,:,2]).item()

                loss = lossx+lossy+lossr

                loss_list.append(loss)

                boardWriter.add_scalar('Loss/test', loss, episode)

                if episode == 0: # save non-trained network 
                    saveONNX(modelx, var_num,reporter, device, episode, onnx_path_x)
                    saveONNX(modely, var_num,reporter, device, episode, onnx_path_y)
                    saveONNX(modelr, var_num,reporter, device, episode, onnx_path_r)

                    reporter.info(f"===== Saved non-trained network")
                elif (loss == min(loss_list) and episode > SAVE_START) or (episode % SAVE_PER_EPISODE == 0) :
                    saveONNX(modelx, var_num,reporter, device, episode, onnx_path_x)
                    saveONNX(modely, var_num,reporter, device, episode, onnx_path_y)
                    saveONNX(modelr, var_num,reporter, device, episode, onnx_path_r)

            # TRAIN REPORT
            if episode % PLOT_PER_EPISODE == 0:
                reporter.info(f"EPISODE {episode}, LOSS {loss}")

                plt.plot(loss_list)
                plt.savefig(fig_path)
                plt.clf()
                # reporter.info(f"LOSS FUNCTION PLOTTED at {fig_path}")

            # EPOCH TRAIN
            random_pick = np.random.choice(len(train_dataset), BATCH_SIZE)
            sample = train_dataset[random_pick,:]
            target = sample[:,0:3]
            input_data = sample[:,3:]

            analystic_target = F(input_data)
            target = target-analystic_target

            X_list = np2tensor(input_data, device)    
            target = np2tensor(target, device)

            # X_list = torch.nn.functional.normalize(X_list, dim=2)

            for _ in range(EPOCH):
                predx = modelx(X_list)
                predy = modely(X_list)
                predr = modelr(X_list)
                
                lossx = loss_fn(predx[:,:,0],target[:,:,0])
                lossy = loss_fn(predy[:,:,0],target[:,:,1])
                lossr = loss_fn(predr[:,:,0],target[:,:,2])


                optimizerx.zero_grad()
                lossx.backward()
                optimizerx.step()

                optimizery.zero_grad()
                lossy.backward()
                optimizery.step()

                optimizerr.zero_grad()
                lossr.backward()
                optimizerr.step()

    finally:
        reporter.info(f"train finished. \nfinal loss: {loss}")
        saveONNX(modelx, var_num,reporter, device, "FINAL" + str(EPISODE), onnx_path_x)
        saveONNX(modely, var_num,reporter, device, "FINAL" + str(EPISODE), onnx_path_y)
        saveONNX(modelr, var_num,reporter, device, "FINAL" + str(EPISODE), onnx_path_r)        

if __name__ == '__main__':
    main()

# def dFdX(sample):
#     x_dot = sample[:,0]
#     y_dot = sample[:,1]    
#     yaw_dot = sample[:,2]
#     delta = sample[:,3]
#     # Frl = sample[:,4]
#     # Frr = sample[:,5]
    
#     dfdx_op = np.array([
#         [                                               -(2*Ca*np.sin(delta)*(y_dot + lf*yaw_dot))/(m*x_dot**2),             yaw_dot + (2*Ca*np.sin(delta))/(m*x_dot),                   y_dot + (2*Ca*lf*np.sin(delta))/(m*x_dot)],
#         [((2*Ca*(y_dot - lr*yaw_dot))/x_dot**2 + (2*Ca*np.cos(delta)*(y_dot + lf*yaw_dot))/x_dot**2)/m - yaw_dot,       -((2*Ca)/x_dot + (2*Ca*np.cos(delta))/x_dot)/m, ((2*Ca*lr)/x_dot - (2*Ca*lf*np.cos(delta))/x_dot)/m - x_dot],
#         [  -((2*Ca*lr*(y_dot - lr*yaw_dot))/x_dot**2 - (2*Ca*lf*np.cos(delta)*(y_dot + lf*yaw_dot))/x_dot**2)/Iz, ((2*Ca*lr)/x_dot - (2*Ca*lf*np.cos(delta))/x_dot)/Iz,   -((2*Ca*np.cos(delta)*lf**2)/x_dot + (2*Ca*lr**2)/x_dot)/Iz]
#     ])
    
#     dfdu_op = np.array([ 
#         [      -(2*Ca*np.sin(delta) + 2*Ca*np.cos(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/m,   1/m *(x_dot/x_dot),  1/m *(x_dot/x_dot)],
#         [       (2*Ca*np.cos(delta) - 2*Ca*np.sin(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/m,     0 *(x_dot/x_dot),    0 *(x_dot/x_dot)],
#         [(2*Ca*lf*np.cos(delta) - 2*Ca*lf*np.sin(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/Iz, -w/Iz *(x_dot/x_dot), w/Iz *(x_dot/x_dot)]
#     ])
    
#     dfdx = np.concatenate((dfdx_op, dfdu_op), axis=1) # state_num * (state_num + control_num)
    
#     dfdx = np.swapaxes(dfdx, 0, 2)
#     dfdx = np.swapaxes(dfdx, 1, 2)    
#     return dfdx