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

# START ========================================
# CONSTANTS ====================================
EPISODE = int(1e3)
EPOCH = int(1)
BATCH_SIZE = 32
VALIDATION_SIZE = 16
SAVE_START = int(1)
SAVE_PER_EPISODE = int(1e1)
PLOT_PER_EPISODE = int(10)
PLOT_FROM = 1e1

SPECIAL_COMMENT = " \
    Trained with myPDE, G, Again with save onnx per 10; 128 nodes\
"

# END ==========================================
# CONSTANTS ====================================

def G(sample):
    x1 = sample[:,0]
    x2 = sample[:,1]    
    x3 = sample[:,2]
    u1 = sample[:,3]
    u2 = sample[:,4]
    u3 = sample[:,5]
    
    x1_ddot = ((np.cos(u1) - (x2+x3) * np.sin(u1)) + u2+u3) + x3*x2
    x2_ddot = ((np.sin(u1) + (x2+x3) * np.cos(u1)) + (x2-x3)) - x3*x1
    x3_ddot = (((np.sin(u1) + (x2+x3) * np.cos(u1)) - (x2-x3)) + u3-u2)

    ddots = np.array([x1_ddot, x2_ddot, x3_ddot])
    ddots = ddots.T
    return ddots 

def F(sample):
    x1 = sample[:,0]
    x2 = sample[:,1]    
    x3 = sample[:,2]
    u1 = sample[:,3]
    u2 = sample[:,4]
    u3 = sample[:,5]
    
    x1_ddot = (((np.cos(u1) - (x2+x3) * np.sin(u1)) + u2+u3) + x3*x2) * 10
    x2_ddot = (((np.sin(u1) + (x2+x3) * np.cos(u1)) + (x2-x3)) - x3*x1) * 10
    x3_ddot = ((((np.sin(u1) + (x2+x3) * np.cos(u1)) - (x2-x3)) + u3-u2))  * 10

    ddots = np.array([x1_ddot, x2_ddot, x3_ddot])
    ddots = ddots.T
    return ddots 

def main():  
    np.random.seed(0)
    # reporter config
    # cur_dir = os.getcwd()
    cur_time = strftime("%m%d_%I%M%p", localtime(time()))
    cur_time = cur_time + "_MY_PDE"
    log_name = cur_time
    board_name = "runs/" + cur_time
    reporter = reporter_loader("info", log_name)
    boardWriter = SummaryWriter(board_name)

    reporter.info("***** SPECIAL COMMENT *****")
    reporter.info(f"{SPECIAL_COMMENT}")

    reporter.info(f"Using random data")
    
    var_num = 6

    # Trainer Setting
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NN(var_num).to(device)
    reporter.info(summary(model, (1,var_num)))
    loss_fn = nn.MSELoss()
    reporter.info(f"Device({device}) is working for train")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []

    # result path
    onnx_path = "./savemodel/"  + cur_time + "/"
    fig_path = "./fig/" + cur_time + "loss.png"

    try:
        os.mkdir(onnx_path)
    except:
        os.rmdir(onnx_path)
        os.mkdir(onnx_path)

    try: 
        reporter.info(f"Train Started")
        for episode in range(EPISODE):
            with torch.no_grad():
                input_data = (np.random.rand(VALIDATION_SIZE, var_num) - 1/2) * 2 * 1.5
                target = G(input_data) #+ F(input_data)

                X_list = np2tensor(input_data, device)    
                target = np2tensor(target, device)

                pred = model(X_list)
                loss = loss_fn(pred,target).item()
                loss_list.append(loss)

                boardWriter.add_scalar('Loss/test', loss, episode)

                if episode == 0: # save non-trained network 
                    saveONNX(model, var_num,reporter, device, episode, onnx_path)
                    reporter.info(f"===== Saved non-trained network")
                elif (loss == min(loss_list) and episode > SAVE_START) or (episode % SAVE_PER_EPISODE == 0) :
                    saveONNX(model,var_num, reporter, device, episode, onnx_path)

            # TRAIN REPORT
            if episode % PLOT_PER_EPISODE == 0:
                reporter.info(f"EPISODE {episode}, LOSS {loss}")

                plt.plot(loss_list)
                plt.savefig(fig_path)
                plt.clf()
                # reporter.info(f"LOSS FUNCTION PLOTTED at {fig_path}")

            # EPOCH TRAIN
            input_data = (np.random.rand(BATCH_SIZE, var_num) - 1/2) * 2 * 1.5
            target = G(input_data)# + F(input_data)

            X_list = np2tensor(input_data, device)    
            target = np2tensor(target, device)

            for _ in range(EPOCH):
                pred = model(X_list)
                loss = loss_fn(pred,target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    finally:
        reporter.info(f"train finished. \nfinal loss: {loss} at episode {episode}")
        saveONNX(model,var_num,reporter, device, "FINAL", onnx_path)

if __name__ == '__main__':
    main()
