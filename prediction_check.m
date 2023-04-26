clear
close all
%% constants
file_name = "0421_0641PM0";
nn_name = "0426_0243PM/20";
TEST_TRAIN_DATA_RATE = 0.1;
test_num = 10;
    Ts = 0.01;
Np = 5; Nc = 1;
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
CM_data = CM_data(:,2:end);
trg = CM_data(:, 1:3)';
CM_data = CM_data(:, 4:end);

prediction_list = zeros(test_num, Np); 
prediction_analytic_list = zeros(test_num, Np); 
err_list = zeros(test_num, 1);

CM_data = CM_data';
q = 0;
for sample = CM_data
    f0 = analy_F(sample);
    dfdx0 = analy_dFdX(sample);
    Dx = f0 + dfdx0 * sample;
    Dx = Dx * Ts;

    input_sample = dlarray(sample, "CB");

    g = zeros(3,6);
    for i = 1:1:3
        [v,grad] = dlfeval(@model,nn,input_sample, i);
        tmp = extractdata(grad); % input_sample = extractdata(input_sample);
        g(i,:) = tmp;
    end

    state = sample(1:3);
    controlInput = repmat(sample(4:6), Nc, 1);

    [Phi, F, gamma] = predmat(dfdx0+g);
    traj = F*state + Phi * controlInput + gamma * Dx;
    traj = reshape(traj, 3, []);
        prediction_list((q+1)*3:(q+1)*3+2,:) = traj;
    
    [Phi, F, gamma]  = predmat(dfdx0);
    traj = F*state + Phi * controlInput + gamma * Dx;
    traj = reshape(traj, 3, []);
        prediction_analytic_list((q+1)*3:(q+1)*3+2,:) = traj;
        q = q+1;

    err = (trg(:,q) - f0) - extractdata(predict(nn, input_sample));
    err_list(q) = sqrt(sum(err.^2));
end

%% plot
for s = 1:1:q
    for p = 1:1:3
        figure(s)
        
        subplot(3,1,1)
        plot(prediction_list(s,:)*3.6, 'r');
        hold on 
        plot(prediction_analytic_list(s,:)*3.6);
        subplot(3,1,2)
        plot(prediction_list(s+1,:)*3.6, 'r');
        hold on 
        plot(prediction_analytic_list(s+1,:)*3.6);        
        subplot(3,1,3)
        plot(prediction_list(s+2,:)*180/pi, 'r');
        hold on 
        plot(prediction_analytic_list(s+2,:)*180/pi); 
        legend
        % title("Target - F - G")
        % xlabel("") 
        % ylabel("X2")
    end
end
err_list
%%
function f = analy_F(sample)
    Ca = 756.349/(0.6*pi/180);
    m = 1644.80;
    Iz = 2488.892;
    lf = 1.240;
    lr = 1.510;
    w = 0.8;
    

    vx = sample(1);
    vy = sample(2) ;   
    yawRate = sample(3);
    Frl = sample(5);
    Frr = sample(6);
    StrAng = sample(4);

    Fxf = 0;
    Fyf = 2 * Ca * (StrAng - ((vy+lf*yawRate)/ vx));
    Fyr = 2 * Ca * (       - ((vy-lr*yawRate)/ vx));

    del_Fxf = 0;
    del_Fxr = Frr - Frl;

    x_ddot = ((Fxf * cos(StrAng) - Fyf * sin(StrAng)) + Frl+Frr) * 1/m + yawRate*vy;
    y_ddot = ((Fxf * sin(StrAng) + Fyf * cos(StrAng)) + Fyr) * 1/m - yawRate*vx;
    psi_ddot = ((lf * (Fxf * sin(StrAng) + Fyf * cos(StrAng)) - lr * Fyr) + w * (del_Fxf + del_Fxr)) / Iz;

    f = [x_ddot; y_ddot; psi_ddot];
end
%% dFdX
function dfdx = analy_dFdX(sample)
%     Cf = 435.418/0.296296;
    Cr = 756.349/(0.6*pi/180);
    Cf = Cr;
    m = 1644.80;
    Iz = 2488.892;
    lf = 1.240;
    lr = 1.510;
    w = 0.8;
    
    x_dot = sample(1);
    y_dot = sample(2) ;   
    psi_dot = sample(3);
    % Frl = sample(:,4);
    % Frr = sample(:,5);
    delta = sample(4);

    dfdx_op = [
 
[                                               -(2*Cf*sin(delta)*(y_dot + lf*psi_dot))/(m*x_dot^2),             psi_dot + (2*Cf*sin(delta))/(m*x_dot),                   y_dot + (2*Cf*lf*sin(delta))/(m*x_dot)]
[((2*Cr*(y_dot - lr*psi_dot))/x_dot^2 + (2*Cf*cos(delta)*(y_dot + lf*psi_dot))/x_dot^2)/m - psi_dot,       -((2*Cr)/x_dot + (2*Cf*cos(delta))/x_dot)/m, ((2*Cr*lr)/x_dot - (2*Cf*lf*cos(delta))/x_dot)/m - x_dot]
[  -((2*Cr*lr*(y_dot - lr*psi_dot))/x_dot^2 - (2*Cf*lf*cos(delta)*(y_dot + lf*psi_dot))/x_dot^2)/Iz, ((2*Cr*lr)/x_dot - (2*Cf*lf*cos(delta))/x_dot)/Iz,   -((2*Cf*cos(delta)*lf^2)/x_dot + (2*Cr*lr^2)/x_dot)/Iz]
 
 ];

    dfdu_op = [
        
[      -(2*Cf*sin(delta) + 2*Cf*cos(delta)*(delta - (y_dot + lf*psi_dot)/x_dot))/m,   1/m,  1/m]
[       (2*Cf*cos(delta) - 2*Cf*sin(delta)*(delta - (y_dot + lf*psi_dot)/x_dot))/m,     0,    0]
[(2*Cf*lf*cos(delta) - 2*Cf*lf*sin(delta)*(delta - (y_dot + lf*psi_dot)/x_dot))/Iz, -w/Iz, w/Iz]
 
 ];
    
    dfdx = [dfdx_op dfdu_op];
end

%% gradient model
function [y, g] = model(net, x, i)
   y = forward(net, x);
   % g = dlgradient(y, net.Learnables);
   y = y(i);
   g = dlgradient(y, x);
end

%% predicitive matrices
function [Phi, F, gamma] = predmat(h)
    Ts = 0.01;
Np = 5; Nc = 1;

    A = h(:,1:3);
    B = h(:,4:6);
    C = eye(3);
    
    [stateSize, inputSize] = size(C);
    predStateSize = stateSize * Np;
    predInputSize = inputSize * Nc;

    A = A*Ts+eye(stateSize);
    B = B*Ts;

    F = ones(predStateSize, stateSize);
    Phi = zeros(predStateSize, predInputSize);
    gamma = zeros(predStateSize, stateSize);

    pre_gamma = zeros(3,3);
    for i = 1:1:Np
        F((i-1)*stateSize+1:i*stateSize, ...
            1:stateSize) = C * A^i;
        gamma((i-1)*stateSize+1:i*stateSize, ...
            1:stateSize) = A^(i-1) + pre_gamma;
        pre_gamma = gamma((i-1)*stateSize+1:i*stateSize, 1:stateSize);
        for j = 1:1:Nc
            tmp =  C * A^(i-1) * B;
            Phi((i+j-2)*stateSize+1:(i+j-1)*stateSize, ...
                (j-1)*inputSize+1:j*inputSize) = tmp;
        end
    end
    
    Phi = Phi(1:predStateSize,1:predInputSize);
end
