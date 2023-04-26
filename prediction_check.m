clear
close all
rng('shuffle')
format shortEng
format compact
%% constants
file_name = "0421_0641PM0";
nn_name = "0426_0243PM/20";
TEST_TRAIN_DATA_RATE = 0.1;
test_num = 5;
Ts = 0.01; Np = 20; Nc = Np;
PLOT_DATA = false;
%%
nn = "./savemodel/" + nn_name + ".onnx";
nn = importONNXNetwork( ...
  nn,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
);
% analyzeNetwork(nn)

%% data fron csv
shuffled_file_name = "processed_csv_data/shuffled_" + file_name +".csv";
raw_file_name = "processed_csv_data/" +  file_name + ".csv";
CM_data = csvread(shuffled_file_name);
raw_data = csvread(raw_file_name);
[sample_num, var_num] = size(CM_data);

CM_data = CM_data(1:floor(sample_num*TEST_TRAIN_DATA_RATE), :);
CM_data = CM_data(randperm(test_num),:);

%% target, prediction calc
raw_num = CM_data(:,1);
CM_data = CM_data(:,2:end);
trg = CM_data(:, 1:3)';
CM_data = CM_data(:, 4:end);

prediction_list = zeros(Np, 3*test_num); 
prediction_analytic_list = zeros(Np, 3*test_num);
raw_state_list = zeros(Np, 3*test_num); 
err_list = zeros(test_num, 3);

CM_data = CM_data';
q = 0;
for sample = CM_data
    f0 = analy_F(sample);
    dfdx0 = analy_dFdX(sample);
    Dx = f0 + dfdx0 * sample;
    Dx = Dx * Ts;

    input_sample = dlarray(sample, "CB");
    
    % prediction test
    err = (trg(:,q+1) - f0) - extractdata(predict(nn, input_sample));
    err_list(q+1, 1) = sqrt(sum(err.^2));  

    g = zeros(3,6);
    for i = 1:1:3
        [v,grad] = dlfeval(@model,nn,input_sample, i);
        tmp = extractdata(grad); % input_sample = extractdata(input_sample);
        g(i,:) = tmp;
    end

    state = sample(1:3);

    raw_traj = raw_data(raw_num(q+1):raw_num(q+1)+Np-1, 2:end);
    raw_state = raw_traj(:,4:6);
    raw_state_list(:, (q)*3+1:(q)*3+3) = raw_state;

    raw_control = raw_traj(:,7:9);
    controlInput = reshape(raw_control, [], 1);

    [Phi, F, gamma] = predmat(dfdx0+g, Np, Nc, Ts);
    traj = F*state + Phi * controlInput + gamma * Dx;
    traj = reshape(traj, [], 3);
    prediction_list(:, (q)*3+1:(q)*3+3) = traj;
    
    err = raw_state - traj;
    % err_list(q, 2) = sqrt(sum(err.^2));  
    err_list(q+1, 2) = norm(err, 2);

    [Phi, F, gamma]  = predmat(dfdx0, Np, Nc, Ts);
    analytic_traj = F*state + Phi * controlInput + gamma * Dx;
    analytic_traj = reshape(analytic_traj, [], 3);
    prediction_analytic_list(:, (q)*3+1:(q)*3+3) = analytic_traj;

    err = raw_state - analytic_traj;
    % err_list(q, 2) = sqrt(sum(err.^2));  
    err_list(q+1, 3) = norm(err, 2);

    q = q+1;
end

%% plot
if PLOT_DATA
    for s = 1:1:test_num
        figure(s)
        
        subplot(3,1,1)
        plot(prediction_list(:, (s-1)*3+1)*3.6, 'r');
        hold on 
        plot(prediction_analytic_list(:, (s-1)*3+1)*3.6, 'b');
        plot(raw_state_list(:, (s-1)*3+1)*3.6, 'g');
        subplot(3,1,2)
        plot(prediction_list(:, (s-1)*3+2)*3.6, 'r');
        hold on 
        plot(prediction_analytic_list(:, (s-1)*3+2)*3.6, 'b');   
        plot(raw_state_list(:, (s-1)*3+2)*3.6, 'g');     
        subplot(3,1,3)
        plot(prediction_list(:, (s-1)*3+3)*180/pi, 'r');
        hold on 
        plot(prediction_analytic_list(:, (s-1)*3+3)*180/pi, 'b'); 
        plot(raw_state_list(:, (s-1)*3+3)*180/pi, 'g');

        % legend
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
function [Phi, F, gamma] = predmat(h, Np, Nc, Ts)
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
