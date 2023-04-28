function report_list = prediction_check(PLOT_DATA, seed, NN_NAME, FILE_NAME, TEST_NUM, Ts, Np)
    %% constants
    if nargin == 3
        FILE_NAME = "0421_0641PM0";

        TEST_NUM = 5;
        Ts = 0.01; Np = 20; Nc = Np;
        TEST_TRAIN_DATA_RATE = 0.1;
    elseif nargin < 5
        seed = rng("Shuffle").Seed;
        NN_NAME = "0427_0702PM/18";

        FILE_NAME = "0421_0641PM0";

        TEST_NUM = 5;
        Ts = 0.01; Np = 20; Nc = Np;
        TEST_TRAIN_DATA_RATE = 0.1;
        
        PLOT_DATA = false;
    end
    %%
    fprintf("NN_NAME: %s\n", NN_NAME)
    %% simulation constants
    state_num = 3;
    control_num = 3;
    %% prediction test constants
    rng(seed)
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
    
    test_data_index = randi(floor(sample_num*TEST_TRAIN_DATA_RATE), 1, TEST_NUM);
    shuffled_CM_data = shuffled_CM_data(1:floor(sample_num*TEST_TRAIN_DATA_RATE), :);
    shuffled_CM_data = shuffled_CM_data(test_data_index,:);
    
    %% target, prediction calc
    ori_index = shuffled_CM_data(:,1);
    shuffled_CM_data = shuffled_CM_data(:,2:end);
    err_target = shuffled_CM_data(:, 1:3)';
    shuffled_CM_data = shuffled_CM_data(:, 4:end);
    shuffled_CM_data = shuffled_CM_data';

    % pre-allocate matrices
    prediction_list = zeros(1+Np, state_num*TEST_NUM); 
    prediction_nominal_list = zeros(1+Np, state_num*TEST_NUM); 
    prediction_analytic_list = zeros(1+Np, state_num*TEST_NUM);
    ori_state_list = zeros(1+Np, state_num*TEST_NUM); 
    report_list = zeros(TEST_NUM, 5); 
    control_list = zeros(Np, control_num*TEST_NUM);
    
    % prediction start
    q = 1;
    for sample = shuffled_CM_data
        f0 = analy_F(sample);
        dfdx0 = analy_dFdX(sample);
      
    
        input_sample = dlarray(sample, "CB");
        
        % NN Uncertainty Prediction Test =======================================
        prediction_err = (err_target(:,q) - f0) - extractdata(predict(nn, input_sample));
        report_list(q, 2) = norm(prediction_err, 2);  
    
        % NN Gradient Calc =====================================================
        g = zeros(3,6);
        for i = 1:1:3
            [v,grad] = dlfeval(@model,nn,input_sample, i);
            tmp = extractdata(grad); % input_sample = extractdata(input_sample);
            g(i,:) = tmp;
        end
        g0 = extractdata(predict(nn, input_sample));
    
        cur_state = sample(1:3);
    
        ori_traj = CM_data(ori_index(q):ori_index(q)+Np-1, 2:end);
        ori_state = ori_traj(:,4:6);
        ori_state_list(:, (q-1)*3+1:(q-1)*3+3) = [cur_state'; ori_state];
        report_list(q, 1) = ori_state(1);

        ori_control = ori_traj(:,7:9);
        control_list(:, (q-1)*3+1:(q-1)*3+3) = ori_control;
        ori_control = ori_control';
        controlInput = reshape(ori_control, [], 1);
    
        % Prediction Proposed ==================================================
        Dx = f0 + dfdx0 * sample + g0 + g * sample;
        Dx = Dx * Ts;    
    
        [Phi, F, gamma] = predmat(dfdx0+g, Np, Nc, Ts);
        traj = F*cur_state + Phi * controlInput + gamma * Dx;
        traj = reshape(traj, 3, []);
        traj = traj';
        prediction_list(:, (q-1)*3+1:(q-1)*3+3) = [cur_state'; traj];
    
        total_prediction_err = ori_state - traj;
        report_list(q, 3) = norm(total_prediction_err, 2);
    
        % Prediction Nominal ==================================================
        Dx = f0 + dfdx0 * sample + g0;
        Dx = Dx * Ts;    
    
        [Phi, F, gamma] = predmat(dfdx0, Np, Nc, Ts);
        traj = F*cur_state + Phi * controlInput + gamma * Dx;
        traj = reshape(traj, 3, []);
        traj = traj';
        prediction_nominal_list(:, (q-1)*3+1:(q-1)*3+3) = [cur_state'; traj];
    
        norm_prediction_err = ori_state - traj;
        report_list(q, 4) = norm(norm_prediction_err, 2);
       
        % Prediction Analytic ==================================================
        Dx_analytic = f0 + dfdx0 * sample;
        Dx_analytic = Dx_analytic * Ts;  
    
        [Phi, F, gamma]  = predmat(dfdx0, Np, Nc, Ts);
        analytic_traj = F*cur_state + Phi * controlInput + gamma * Dx_analytic;
        analytic_traj = reshape(analytic_traj, 3, []);
        analytic_traj = analytic_traj';
        prediction_analytic_list(:, (q-1)*3+1:(q-1)*3+3) = [cur_state'; analytic_traj];
    
        analytic_err = ori_state - analytic_traj;
        report_list(q, 5) = norm(analytic_err, 2);
    
        q = q+1;
    end
    
    %% plot and report
    report_list = array2table(report_list, 'VariableNames', ...
        {'Vx0', 'pred_err', 'proposed', 'nominal', 'analytic'});

    x_axis = 0:1:20;
    if PLOT_DATA
        for s = 1:1:TEST_NUM
            figure(s)
            tiledlayout(3,1);
    
            nexttile
            plot(x_axis, prediction_list(:, (s-1)*3+1)*3.6, 'r');
            hold on 
            plot(x_axis, prediction_nominal_list(:, (s-1)*3+1)*3.6, 'k');        
            plot(x_axis, prediction_analytic_list(:, (s-1)*3+1)*3.6, 'b');
            plot(x_axis, ori_state_list(:, (s-1)*3+1)*3.6, 'g');
            xlabel("Time Step [0.01ms]")
            ylabel("Longitudinal Velocity [km/h]") 
            grid on
        
            nexttile
            plot(x_axis, prediction_list(:, (s-1)*3+2)*3.6, 'r');
            hold on 
            plot(x_axis, prediction_nominal_list(:, (s-1)*3+2)*3.6, 'k');   
            plot(x_axis, prediction_analytic_list(:, (s-1)*3+2)*3.6, 'b');   
            plot(x_axis, ori_state_list(:, (s-1)*3+2)*3.6, 'g');     
            xlabel("Time Step [0.01ms]")
            ylabel("Lateral Velocity [km/h]") 
            grid on
        
            nexttile
            plot(x_axis, prediction_list(:, (s-1)*3+3)*180/pi, 'r');
            hold on
            plot(x_axis, prediction_nominal_list(:, (s-1)*3+3)*180/pi, 'k');         
            plot(x_axis, prediction_analytic_list(:, (s-1)*3+3)*180/pi, 'b'); 
            plot(x_axis, ori_state_list(:, (s-1)*3+3)*180/pi, 'g');
            xlabel("Time Step [0.01ms]")
            ylabel("Yaw Rate [deg/s]") 
            grid on
    
            lgd = legend('proposed prediction', 'nominal prediction', 'analytic prediction', 'ground truth');
            lgd.Layout.Tile = 'south';
            lgd.NumColumns = 3;
        end
    end
    
    %% F; analytic system model
    function f = analy_F(sample)
        Ca = 756.349/(0.6*pi/180);
        m = 1644.80;
        Iz = 2488.892;
        lf = 1.240;
        lr = 1.510;
        w = 0.8;
    
        x_dot = sample(1);
        y_dot = sample(2) ;   
        yaw_dot = sample(3);
        delta = sample(4);
        Frl = sample(5);
        Frr = sample(6);
    
        Fxf = 0;
        Fyf = 2 * Ca * (delta - ((y_dot+lf*yaw_dot)/ x_dot));
        Fyr = 2 * Ca * (       - ((y_dot-lr*yaw_dot)/ x_dot));
    
        del_Fxf = 0;
        del_Fxr = Frr - Frl;
    
        x_ddot = ((Fxf * cos(delta) - Fyf * sin(delta)) + Frl+Frr) * 1/m + yaw_dot*y_dot;
        y_ddot = ((Fxf * sin(delta) + Fyf * cos(delta)) + Fyr) * 1/m - yaw_dot*x_dot;
        psi_ddot = ((lf * (Fxf * sin(delta) + Fyf * cos(delta)) - lr * Fyr) + w * (del_Fxf + del_Fxr)) / Iz;
    
        f = [x_ddot; y_ddot; psi_ddot];
    end
    %% dFdX; analytic gradient
    function dfdx = analy_dFdX(sample)
        Ca = 756.349/(0.6*pi/180);
        m = 1644.80;
        Iz = 2488.892;
        lf = 1.240;
        lr = 1.510;
        w = 0.8;
        
        x_dot = sample(1);
        y_dot = sample(2) ;   
        yaw_dot = sample(3);
        delta = sample(4);        
        % Frl = sample(:,4);
        % Frr = sample(:,5);
        
        dfdx_op = [
            [                                               -(2*Ca*sin(delta)*(y_dot + lf*yaw_dot))/(m*x_dot^2),             yaw_dot + (2*Ca*sin(delta))/(m*x_dot),                   y_dot + (2*Ca*lf*sin(delta))/(m*x_dot)]
            [((2*Ca*(y_dot - lr*yaw_dot))/x_dot^2 + (2*Ca*cos(delta)*(y_dot + lf*yaw_dot))/x_dot^2)/m - yaw_dot,       -((2*Ca)/x_dot + (2*Ca*cos(delta))/x_dot)/m, ((2*Ca*lr)/x_dot - (2*Ca*lf*cos(delta))/x_dot)/m - x_dot]
            [  -((2*Ca*lr*(y_dot - lr*yaw_dot))/x_dot^2 - (2*Ca*lf*cos(delta)*(y_dot + lf*yaw_dot))/x_dot^2)/Iz, ((2*Ca*lr)/x_dot - (2*Ca*lf*cos(delta))/x_dot)/Iz,   -((2*Ca*cos(delta)*lf^2)/x_dot + (2*Ca*lr^2)/x_dot)/Iz]
        ];
        
        dfdu_op = [ 
            [      -(2*Ca*sin(delta) + 2*Ca*cos(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/m,   1/m,  1/m]
            [       (2*Ca*cos(delta) - 2*Ca*sin(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/m,     0,    0]
            [(2*Ca*lf*cos(delta) - 2*Ca*lf*sin(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/Iz, -w/Iz, w/Iz]
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
end