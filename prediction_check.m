clear
close all
%% constants
file_name = "0421_0641PM0";
nn_name = "NN_FINAL";
Ts = 0.01;
Np = 20; Nc = 1;
%%
nn = "./savemodel/" + nn_name + ".onnx";
nn = importONNXNetwork( ...
  nn,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
);
% analyzeNetwork(nn)

%% data fron csv
CM_data = data_pre_processing(file_name, false);
CM_data = table2array(CM_data);
CM_data = CM_data(200, :);

[sample_num, var_num] = size(CM_data);
%% target, prediction calc
trg = CM_data(:, 1);
CM_data = CM_data(:, 2:end);
% anal_trg = F(CM_data);

prediction_list = zeros(sample_num, Np); 
i = 1;

CM_data = CM_data';
for sample = CM_data
    f = dFdX(sample);

    input_sample = dlarray(sample, "CB");
    [v,g] = dlfeval(@model,nn,input_sample);
    g = extractdata(g); input_sample = extractdata(input_sample);
    
    h = f+g;

    A = h(1:3); B = h(4:6);
    C = [1 0 0];
    
    A = A*Ts+eye(size(A));
    B = B*Ts;
    C = C; 

    inputSize = 3;

    [outputSize, stateSize] = size(C);

    F = ones(outputSize * Np, stateSize);
    Phi = zeros(outputSize * Np, inputSize * Nc);

    for i = 1:1:Np
        F((i-1)*outputSize+1:i*outputSize, ...
            1:stateSize) = C * A^i;
        
        for j = 1:1:Nc
            tmp = C * A^(i-1) * B;
            Phi((i+j-2)*outputSize+1:(i+j-1)*outputSize, ...
                (j-1)*inputSize+1:j*inputSize) = tmp;
        end
    end

    Phi = Phi(1:outputSize * Np,1:inputSize * Nc);

    prediction = F * input_sample(1:3) + Phi * input_sample(4:6)


    prediction_list(i,:) = prediction;
    i = i+1;
end

%% plot
% target plot
figure(1)
plot(prediction_list)
% title("Target - F - G")
% xlabel("") 
% ylabel("X2")

%% analystic jacobian
function f = dFdX(sample)
    Cf = 756.349/(0.6*pi/180);
    lf = 1.240;
    m = 1644.80;

    x_dot = sample(1);
    y_dot = sample(2) ;   
    psi_dot = sample(3);
    % Frl = sample(:,4);
    % Frr = sample(:,5);
    delta = sample(6);

    f = [ ...
    -(2*Cf*sin(delta)*(y_dot + lf*psi_dot))/(m*x_dot^2), psi_dot + (2*Cf*sin(delta))/(m*x_dot), y_dot + (2*Cf*lf*sin(delta))/(m*x_dot), ...
    -(2*Cf*sin(delta) + 2*Cf*cos(delta)*(delta - (y_dot + lf*psi_dot)/x_dot))/m, 1/m, 1/m ...
    ];
    f = f';
end

%% gradient model
function [y, g] = model(net, x)
   y = forward(net, x);
   % g = dlgradient(y, net.Learnables);
   g = dlgradient(y, x);
end

%% analystic PDE
% function x_ddot = F(sample)
%     Ca = 756.349/(0.6*pi/180);
%     l = 1.240;
%     m = 1644.80;
% 
%     vx = sample(:,1);
%     vy = sample(:,2) ;   
%     yawRate = sample(:,3);
%     Frl = sample(:,4);
%     Frr = sample(:,5);
%     StrAng = sample(:,6);
% 
%     Fxf = 0;
% 
%     Fyf = 2 * Ca * (StrAng - ((vy+l*yawRate)./ vx));
% 
%     x_ddot = ((Fxf .* cos(StrAng) - Fyf .* sin(StrAng)) + Frl+Frr) * 1/m + yawRate.*vy;
% end

