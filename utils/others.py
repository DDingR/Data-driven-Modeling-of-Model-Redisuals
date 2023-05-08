import csv
import torch
from torch import nn
import numpy as np

# START =====================================
# NEURAL NETWORK ============================
class NN(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),            
            nn.Linear(1024, 1024),
            nn.ReLU(),                      
            nn.Linear(1024, 3)
        )
    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=-1)
        x = self.fc(x)
        return x

def np2tensor(x, device):
    x = np.array(x)
    if len(x.shape) == 2:
        x = x.reshape(1, x.shape[0], x.shape[1])
    elif len(x.shape) == 1:
        x = x.reshape(1, x.shape[0], 1)
    else:
        raise Exception(f"[ERR] input data is empty")
    x = torch.as_tensor(x, device=device, dtype=torch.float32)
    return x

def saveONNX(model, input_shape, reporter, device, episode, onnx_path):
    onnx_name = onnx_path + str(episode) + ".onnx"
    model.eval()
    dummy_input = torch.randn(1,input_shape,device=device, requires_grad=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_name,
        verbose=False
    )
    reporter.info(f"===== ONNX SAVED at {onnx_name}")
# END =======================================
# NEURAL NETWORK ============================

# START =====================================
# ETC =======================================
def csv2dataset(filename):
    '''
    Input data file has to take csv format.
    It will return array whose shape is
        [1, total length of each data, number of data]
    '''
    data = []
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:  
            data.append(np.float_(row))
    data = np.array(data, dtype=np.float32)
    return data
# END =======================================
# ETC =======================================

