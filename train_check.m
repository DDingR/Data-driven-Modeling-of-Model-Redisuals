clear
close all
%% constants
file_name = "0421_0641PM0";
nn_name = "NN_FINAL";

%%
nn = "./savemodel/" + nn_name + ".onnx";
nn = importONNXNetwork( ...
  nn,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
);
% analyzeNetwork(nn)

%% data fron csv
CM_data = data_pre_processing(file_name);
CM_data = table2array(CM_data);
CM_data = CM_data(100:200, :);

[sample_num, var_num] = size(CM_data);
%% target, prediction calc
trg = CM_data(:, 1);
CM_data = CM_data(:, 2:end);
anal_trg = F(CM_data);

g_list = zeros(sample_num, 1); 
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
function x_ddot = F(sample)
    Ca = 756.349/(0.6*pi/180);
    l = 1.240;
    m = 1644.80;

    vx = sample(:,1);
    vy = sample(:,2) ;   
    yawRate = sample(:,3);
    Frl = sample(:,4);
    Frr = sample(:,5);
    StrAng = sample(:,6);
    
    Fxf = 0;

    Fyf = 2 * Ca * (StrAng - ((vy+l*yawRate)./ vx));
   
    x_ddot = ((Fxf .* cos(StrAng) - Fyf .* sin(StrAng)) + Frl+Frr) * 1/m + yawRate.*vy;
end

