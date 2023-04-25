clear
close all
%% constants
file_name = "0421_0641PM0";
nn_name = "0424_0208PMNN_FINAL";
TEST_TRAIN_DATA_RATE = 0.1;
test_num = 10;
%% simulation constants
Ts = 0.01;
Np = 20; Nc = 1;
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
trg = CM_data(:, 1:3);
CM_data = CM_data(:, 4:end);

prediction_list = zeros(test_num, Np); 
prediction_analytic_list = zeros(test_num, Np); 

CM_data = CM_data';
q = 0;
for sample = CM_data
    f = dFdX(sample);

    input_sample = dlarray(sample, "CB");

    [v,g] = dlfeval(@model,nn,input_sample, 1);
    g1 = extractdata(g); % input_sample = extractdata(input_sample);

    [v,g] = dlfeval(@model,nn,input_sample, 2);
    g2 = extractdata(g); % input_sample = extractdata(input_sample);

    [v,g] = dlfeval(@model,nn,input_sample, 3);
    g3 = extractdata(g); % input_sample = extractdata(input_sample);
    
    g = [g1 g2 g3];
    g = g';

    sample = sample([1 2 3 6 4 5]);
    state = sample(1:3);
    controlInput = repmat(sample(4:6), Nc, 1);


augState = [[0 0 0]'; state];

[Phi, F] = predmat(f+g);
traj = F*augState + Phi * controlInput;
traj = reshape(traj, 3, []);
    prediction_list((q+1)*3:(q+1)*3+2,:) = traj;

[Phi, F]  = predmat(f);
traj = F*augState + Phi * controlInput;
traj = reshape(traj, 3, []);
    prediction_analytic_list((q+1)*3:(q+1)*3+2,:) = traj;
    q = q+1;
end

%% plot
for s = 1:1:q
    for p = 1:1:3
        figure(s)
        legend
        subplot(3,1,1)
        plot(prediction_list(s,:));
        hold on 
        plot(prediction_analytic_list(s,:));
        subplot(3,1,2)
        plot(prediction_list(s+1,:));
        hold on 
        plot(prediction_analytic_list(s+1,:));        
        subplot(3,1,3)
        plot(prediction_list(s+2,:));
        hold on 
        plot(prediction_analytic_list(s+2,:));        
        % title("Target - F - G")
        % xlabel("") 
        % ylabel("X2")
    end
end

%% analystic jacobian
function f = dFdX(sample)
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
    delta = sample(6);

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
    
    f = [dfdx_op dfdu_op];
end

%% gradient model
function [y, g] = model(net, x, i)
   y = forward(net, x);
   % g = dlgradient(y, net.Learnables);
   y = y(i);
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

function [Phi, F] = predmat(h)
    Ts = 0.01;
Np = 20; Nc = 1;


        A = h(:,1:3);
        B = h(:,4:6);
        C = eye(3);
    
    [stateSize, inputSize] = size(C);
augStateSize = stateSize * 2;
predStateSize = stateSize * Np;
predInputSize = inputSize * Nc;
    % Start ========================================================================================
    % [A,B,C,D] = augmentation_ext(A,B,C,D,Ts);
    % ==============================================================================================
    A = A*Ts+eye(stateSize);
    B = B*Ts;
    
    A = [A zeros(inputSize, stateSize);
        C*A eye(stateSize)];
    B = [B; C*B];
    C = [zeros(stateSize, inputSize) eye(stateSize)];
    % End ==========================================================================================
    % [A,B,C,D] = augmentation_ext(A,B,C,D,Ts);
    % ==============================================================================================
    
    % Start ========================================================================================
    % [F, Phi] = predMat(A, B, C, Np, Nc);
    % ==============================================================================================
    F = ones(predStateSize, augStateSize);
    Phi = zeros(predStateSize, predInputSize);
    
    for i = 1:1:Np
        F((i-1)*stateSize+1:i*stateSize, ...
            1:augStateSize) = C * A^i;
        for j = 1:1:Nc
            tmp = C * A^(i-1) * B;
            Phi((i+j-2)*stateSize+1:(i+j-1)*stateSize, ...
                (j-1)*inputSize+1:j*inputSize) = tmp;
        end
    end
    
    Phi = Phi(1:predStateSize,1:predInputSize);
end
