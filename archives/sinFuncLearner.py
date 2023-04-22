import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class NN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

def G(X):
    x1, x2, u = X
    return 1/3*np.sin(x1/3+np.pi)*1/2*np.sin(x2) + 1/5*np.sin(u/3)*x2

def dG(X):
    x1, x2, u = X
    return [
        1/3*1/3*np.cos(x1/3+np.pi)*1/2*np.sin(x2),
        1/3*np.sin(x1/3+np.pi)*1/2*np.cos(x2) + 1/5*np.sin(u/3),
        1/5*1/3*np.cos(u/3)*x2
    ]

def main():
    EPISODE = int(1e4)
    EPOCH = int(1e1)
    BATCH_SIZE = 256
    SAVE_PER_EPISODE = int(EPISODE / 10)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NN().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_list = [] 

    loss_list = []

    for episode in range(EPISODE):

        with torch.no_grad():
            X_list = np.random.randn(1, 32, 3) * 5
            target = G(X_list[0].T)    
            # X_list = np.reshape(X_list, [1, 3, 16])
            X_list = np.array(X_list, dtype=np.float32)
            X_list = torch.as_tensor(X_list, device=device)    
            target = np.array(target, dtype=np.float32)  
            target = np.reshape(target, [1, 32, 1])
            target = torch.as_tensor(target, device=device)     

            pred = model(X_list)
            loss = loss_fn(pred, target).item()
            loss_list.append(loss)
            # print(f"loss: {loss}")

        if episode % SAVE_PER_EPISODE == 0:
            fig_name = './fig/loss.png'
            plt.plot(loss_list)
            plt.savefig(fig_name)
            plt.clf()
            print(f"[INFO] LOSS FUNCTION PLOTTED at {fig_name}")

            onnx_name = './savemodel/NN_EPISODE' + str(episode) + '.onnx'
            model.eval()
            dummy_input = torch.randn(1,3,device=device, requires_grad=True)
            torch.onnx.export(
                model,
                dummy_input,
                onnx_name,
                verbose=False
            )
            print(f"[INFO] ONNX SAVED at {onnx_name}")

            print(f"EPISODE {episode}, LOSS {loss}")


        for _ in range(EPOCH):
            X_list = np.random.randn(1, BATCH_SIZE, 3) * 5
            # dataSet = F(X_list) + G(X_list)

            # train
            target = G(X_list[0].T)
            # X_list = np.reshape(X_list, [1, 3, 64])
            X_list = np.array(X_list, dtype=np.float32)        
            X_list = torch.as_tensor(X_list, device=device)
            target = np.array(target, dtype=np.float32)  
            target = np.reshape(target, [1, BATCH_SIZE, 1])
            target = torch.as_tensor(target, device=device)
            pred = model(X_list)

            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    print(f"[INFO] train finished. \nloss: {loss}")
    
    onnx_name = './savemodel/NN_FINAL.onnx'
    model.eval()
    dummy_input = torch.randn(1,3,device=device, requires_grad=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_name,
        verbose=False
    )
    print(f"[INFO] FINAL ONNX SAVED at {onnx_name}")
    
if __name__ == '__main__':
    main()