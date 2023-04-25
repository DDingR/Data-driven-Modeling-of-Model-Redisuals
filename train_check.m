clear
close all
%% constants
file_name = "0421_0641PM0";
nn_name = "0424_0345PMNN_FINAL";
TEST_TRAIN_DATA_RATE = 0.1;
test_num = 100;
trg_test = 1;
%%
nn = "./savemodel/" + nn_name + ".onnx";
nn = importONNXNetwork( ...
  nn,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
);
% analyzeNetwork(nn)

%% data fron csv
file_name = "processed_csv_data/" + file_name +".csv";
CM_data = csvread(file_name);
[sample_num, var_num] = size(CM_data);

CM_data = CM_data(1:floor(sample_num*TEST_TRAIN_DATA_RATE), :);
CM_data = CM_data(randperm(test_num),:);
%% target, prediction calc
trg = CM_data(:, trg_test);
CM_data = CM_data(:, 4:end);
anal_trg = F(CM_data);
anal_trg = anal_trg(:,trg_test);

g_list = zeros(test_num, 1); 
i = 1;

CM_data = CM_data';
for sample = CM_data
    input_sample = dlarray(sample, "CB");
    [f,g] = dlfeval(@model,nn,input_sample);
    g = extractdata(g); input_sample = extractdata(input_sample);
    tmp = input_sample' * g;
    g_list(i) = tmp;
    i = i+1;
end

err = trg - anal_trg - g_list;
%% plot
% target plot
figure(1)
plot(g_list)
title("Target - F - G")
% xlabel("") 
% ylabel("X2")

%% gradient model
function [y, g] = model(net, x)
   y = forward(net, x);
   % g = dlgradient(y, net.Learnables);
   g = dlgradient(y, x);
end

%% analystic PDE
function [x_ddot, y_ddot, psi_ddot] = F(sample)
    Ca = 756.349/(0.6*pi/180);
    lf = 1.240;
    lr = 1.510;
    w = 0.8;
    m = 1644.80;
    Iz = 2488.892;

    vx = sample(:,1);
    vy = sample(:,2) ;   
    yawRate = sample(:,3);
    StrAng = sample(:,4);    
    Frl = sample(:,5);
    Frr = sample(:,6);
    
    Fxf = 0;
    
    Fyf = 2 * Ca * (StrAng - ((vy+lf*yawRate)./ vx));
    Fyr = 2 * Ca * (       - ((vy-lr*yawRate)./ vx));
    
    del_Fxf = 0;
    del_Fxr = Frr - Frl;
    
    x_ddot = ((Fxf .* cos(StrAng) - Fyf .* sin(StrAng)) + Frl+Frr) * 1/m + yawRate.*vy;
    y_ddot = ((Fxf .* sin(StrAng) + Fyf .* cos(StrAng)) + Fyr) * 1/m + yawRate.*vx;
    psi_ddot = ((lf .* (Fxf .* sin(StrAng) + Fyf .* cos(StrAng)) - lr * Fyr) + w/2 * (del_Fxf + del_Fxr)) / Iz;

end

