clear
close all
%% constants
FILE_NAME = "0421_0641PM0";
NN_NAME = "0426_0243PM/79";

TEST_NUM = 5;
Ts = 0.01; Np = 20; Nc = Np;
TEST_TRAIN_DATA_RATE = 0.1;

PLOT_DATA = true;
%% simulation constants
state_num = 3;
control_num = 3;
%% prediction test constants
rng('shuffle')
format shortEng
format compact
%% network load
nn = "./savemodel/" + NN_NAME + ".onnx";
nn = importONNXNetwork( ...
  nn,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
);
% analyzeNetwork(nn)

%% data load
shuffled_file_name = "processed_csv_data/shuffled_" + FILE_NAME +".csv";
ori_file_name = "processed_csv_data/" +  FILE_NAME + ".csv";
shuffled_CM_data = csvread(shuffled_file_name);
CM_data = csvread(ori_file_name);
[sample_num, var_num] = size(shuffled_CM_data);

test_data_index = randi(floor(sample_num*TEST_TRAIN_DATA_RATE), 1, TEST_NUM)
shuffled_CM_data = shuffled_CM_data(1:floor(sample_num*TEST_TRAIN_DATA_RATE), :);
shuffled_CM_data = shuffled_CM_data(test_data_index,:);

%% target, prediction calc
ori_index = shuffled_CM_data(:,1);
shuffled_CM_data = shuffled_CM_data(:,2:end);
err_target = shuffled_CM_data(:, 1:3)';
shuffled_CM_data = shuffled_CM_data(:, 4:end);

prediction_list = zeros(1+Np, state_num*TEST_NUM); 
prediction_analytic_list = zeros(1+Np, state_num*TEST_NUM);
ori_state_list = zeros(1+Np, state_num*TEST_NUM); 
err_list = zeros(TEST_NUM, state_num); 
control_list = zeros(Np, control_num*TEST_NUM);

shuffled_CM_data = shuffled_CM_data';
q = 1;
for sample = shuffled_CM_data
    f0 = analy_F(sample);
    dfdx0 = analy_dFdX(sample);
    Dx = f0 + dfdx0 * sample;
    Dx = Dx * Ts;

    input_sample = dlarray(sample, "CB");
    
    % NN Uncertainty Prediction Test =======================================
    prediction_err = (err_target(:,q) - f0) - extractdata(predict(nn, input_sample));
    err_list(q, 1) = norm(prediction_err, 2);  

    % NN Gradient Calc =====================================================
    g = zeros(3,6);
    for i = 1:1:3
        [v,grad] = dlfeval(@model,nn,input_sample, i);
        tmp = extractdata(grad); % input_sample = extractdata(input_sample);
        g(i,:) = tmp;
    end
    

    cur_state = sample(1:3);
    cur_state(1)

    ori_traj = CM_data(ori_index(q):ori_index(q)+Np-1, 2:end);
    ori_state = ori_traj(:,4:6);
    ori_state_list(:, (q-1)*3+1:(q-1)*3+3) = [cur_state'; ori_state];

    raw_control = ori_traj(:,7:9);
    control_list(:, (q-1)*3+1:(q-1)*3+3) = raw_control;
    raw_control = raw_control';
    controlInput = reshape(raw_control, [], 1);

    [Phi, F, gamma] = predmat(dfdx0+g, Np, Nc, Ts);
    traj = F*cur_state + Phi * controlInput + gamma * Dx;
    traj = reshape(traj, 3, []);
    traj = traj';
    prediction_list(:, (q-1)*3+1:(q-1)*3+3) = [cur_state'; traj];
    
    total_prediction_err = ori_state - traj;
    err_list(q, 2) = norm(total_prediction_err, 2);

    [Phi, F, gamma]  = predmat(dfdx0, Np, Nc, Ts);
    analytic_traj = F*cur_state + Phi * controlInput + gamma * Dx;
    analytic_traj = reshape(analytic_traj, 3, []);
    analytic_traj = analytic_traj';
    prediction_analytic_list(:, (q-1)*3+1:(q-1)*3+3) = [cur_state'; analytic_traj];

    analytic_err = ori_state - analytic_traj;
    err_list(q, 3) = norm(analytic_err, 2);

    q = q+1;
end

%% plot
x_axis = 0:1:20;
if PLOT_DATA
    for s = 1:1:TEST_NUM
        figure(s)
        tiledlayout(3,1);

        nexttile
        plot(x_axis, prediction_list(:, (s-1)*3+1)*3.6, 'r');
        hold on 
        plot(x_axis, prediction_analytic_list(:, (s-1)*3+1)*3.6, 'b');
        plot(x_axis, ori_state_list(:, (s-1)*3+1)*3.6, 'g');
        xlabel("Time Step [0.01ms]")
        ylabel("Longitudinal Velocity [km/h]") 
        grid on
    
        nexttile
        plot(x_axis, prediction_list(:, (s-1)*3+2)*3.6, 'r');
        hold on 
        plot(x_axis, prediction_analytic_list(:, (s-1)*3+2)*3.6, 'b');   
        plot(x_axis, ori_state_list(:, (s-1)*3+2)*3.6, 'g');     
        xlabel("Time Step [0.01ms]")
        ylabel("Lateral Velocity [km/h]") 
        grid on
    
        nexttile
        plot(x_axis, prediction_list(:, (s-1)*3+3)*180/pi, 'r');
        hold on
        plot(x_axis, prediction_analytic_list(:, (s-1)*3+3)*180/pi, 'b'); 
        plot(x_axis, ori_state_list(:, (s-1)*3+3)*180/pi, 'g');
        xlabel("Time Step [0.01ms]")
        ylabel("Yaw Rate [deg/s]") 
        grid on

        lgd = legend('proposed prediction', 'analytic prediction', 'ground truth');
        lgd.Layout.Tile = 'south';
        lgd.NumColumns = 3;
    end
end
err_list
%% F; analytic system model
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
%% dFdX; analytic gradient
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

%% function to calculate gradient
function [y, g] = model(net, x, i)
   y = forward(net, x);
   y = y(i);
   g = dlgradient(y, x);
end

%% predicitive matrices; for non-augmented system
function [Phi, F, gamma] = predmat(h, Np, Nc, Ts)
% X = F * x_current + Phi * controlSequence + gamma * Dx
%   Dx means linearization constants
%
% these matrices are not for augmented system!!!
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
