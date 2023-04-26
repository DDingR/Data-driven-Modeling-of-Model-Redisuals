import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time, localtime, strftime

from others import *
from reporter import *

raw_csv_dir = "0421_0641PM0"

# START ========================================
# CONSTANTS ====================================
EPISODE = int(1e3)
EPOCH = int(1e5)
BATCH_SIZE = 256
VALIDATION_SIZE = 16
SAVE_START = int(1)
PLOT_PER_EPISODE = int(1)
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

# known function
# sample_data = [ax vx vy yawRate FRL FRR StrAng] ->  7

def F(sample):
    vx = sample[:,0]
    vy = sample[:,1]    
    yawRate = sample[:,2]
    StrAng = sample[:,3]
    Frl = sample[:,4]
    Frr = sample[:,5]
    
    Fxf = 0
    Fyf = 2 * Ca * (StrAng - ((vy+lf*yawRate)/ vx))
    Fyr = 2 * Ca * (       - ((vy-lr*yawRate)/ vx))

    del_Fxf = 0
    del_Fxr = Frr - Frl

    x_ddot = ((Fxf * np.cos(StrAng) - Fyf * np.sin(StrAng)) + Frl+Frr) * 1/m + yawRate*vy
    y_ddot = ((Fxf * np.sin(StrAng) + Fyf * np.cos(StrAng)) + Fyr) * 1/m - yawRate*vx
    psi_ddot = ((lf * (Fxf * np.sin(StrAng) + Fyf * np.cos(StrAng)) - lr * Fyr) + w * (del_Fxf + del_Fxr)) / Iz

    ddots = np.array([x_ddot, y_ddot, psi_ddot])
    ddots = ddots.T
    return ddots 

def main():  
    # reporter config
    cur_dir = os.getcwd()
    cur_time = strftime("%m%d_%I%M%p", localtime(time()))
    log_name = cur_time + ".log"
    reporter = reporter_loader("info", log_name)

    # dataset load
    raw_csv = "processed_csv_data/" + raw_csv_dir +".csv"
    dataset = csv2dataset(raw_csv)
    [sample_num, _] = dataset.shape
    reporter.info(f"Loaded csv data is {raw_csv_dir}")

    test_dataset = dataset[0:int(sample_num*TEST_TRAIN_DATA_RATE), 1:]
    train_dataset = dataset[int(sample_num*TEST_TRAIN_DATA_RATE):, 1:]

    # Trainer Setting
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NN().to(device)
    loss_fn = nn.MSELoss()
    reporter.info(f"Device({device}) is working for train")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []

    # result path
    onnx_path = "./savemodel/"  + cur_time + "/"
    fig_path = "./fig/" + cur_time + "loss.png"

    os.mkdir(onnx_path)

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

                pred = model(X_list)
                # loss = torch.sqrt(loss_fn(pred, target)).item()
                loss = loss_fn(pred,target).item()
                loss_list.append(loss)

                if episode == 0: # save non-trained network 
                    saveONNX(model, device, episode, onnx_path)
                    reporter.info(f"Saved non-trained network")
                elif loss == min(loss_list) and episode > SAVE_START:
                    saveONNX(model, device, episode, onnx_path)

            # TRAIN REPORT
            if episode % PLOT_PER_EPISODE == 0:
                reporter.info(f"EPISODE {episode}, LOSS {loss}")

                plt.plot(loss_list)
                plt.savefig(fig_path)
                plt.clf()
                reporter.info(f"LOSS FUNCTION PLOTTED at {fig_path}")

            # EPOCH TRAIN
            for _ in range(EPOCH):
                random_pick = np.random.choice(len(train_dataset), BATCH_SIZE)
                sample = train_dataset[random_pick,:]
                target = sample[:,0:3]
                input_data = sample[:,3:]
                analystic_target = F(input_data)

                target = target-analystic_target

                X_list = np2tensor(input_data, device)    
                target = np2tensor(target, device)

                pred = model(X_list)
                # loss = torch.sqrt(loss_fn(pred, target))
                loss = loss_fn(pred,target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    finally:
        reporter.info(f"train finished. \nfinal loss: {loss}")
        saveONNX(model, device, "FINAL", onnx_path)

if __name__ == '__main__':
    main()