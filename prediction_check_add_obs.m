function report_list = prediction_check_add_obs(PLOT_DATA, seed, NN_NAME, FILE_NAME, TEST_NUM, Ts, Np)
    close all
    fprintf("\n======== Prediction Test at %s ========\n", char(datetime))
    %% constants
    if nargin == 3
        FILE_NAME = "0501_0133PM";

        TEST_NUM = 5;
        Ts = 0.01; Np = 100; Nc = Np;
    elseif nargin < 5
        seed = rng("Shuffle").Seed;
        NN_NAME = "0504_0941PM/12000";

%         FILE_NAME = "0503_0649PM_withFy";
        FILE_NAME = "0504_0300PM";

         TEST_NUM = 50;
        Ts = 0.01; Np = 20; Nc = Np;

        PLOT_DATA = true;
%         PLOT_DATA = false;
    end

    TEST_TRAIN_DATA_RATE = 0.1; Nc = Np;
    %% overwrite constant!
    seed = 2000;
%     NN_NAME = "0501_1040PM/FINAL";
%     FILE_NAME = "0501_0133PM";
%     TEST_NUM = 5;
%     Ts = 0.01; Np = 100; Nc = Np;
%     TEST_TRAIN_DATA_RATE = 0.1;
%     PLOT_DATA = true;

    if TEST_NUM > 5
        PLOT_DATA = false;
    end

    %% simulation constants
    state_num = 3;
    control_num = 3;
    %% prediction test constants
    fprintf("test seed: %d\n", seed)
    rng(seed)
    format shortEng
    format compact
    %% network load
    fprintf("Loading Neural Network NN_NAME: %s\n", NN_NAME)    
    nn = "./savemodel/" + NN_NAME + ".onnx";
    nn = importONNXNetwork( ...
      nn,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
    );
    % analyzeNetwork(nn)
    
    %% data load
    fprintf("Loading DataSet FILE_NAME: %s\n", FILE_NAME)        
    shuffled_file_name = "processed_csv_data/shuffled_" + FILE_NAME +".csv";
    ori_file_name = "processed_csv_data/" +  FILE_NAME + ".csv";
    shuffled_CM_data = readtable(shuffled_file_name);
    CM_data = readtable(ori_file_name);
    
    shuffled_CM_data = table2array(shuffled_CM_data);
    CM_data = table2array(CM_data);    

    [sample_num, var_num] = size(shuffled_CM_data);

    %% test data index peek
%     test_data_index = randi(floor(sample_num*TEST_TRAIN_DATA_RATE), 1, TEST_NUM);
    test_data_num = floor(sample_num*TEST_TRAIN_DATA_RATE);
    shuffled_CM_data = shuffled_CM_data(1:test_data_num, :);
%     shuffled_CM_data = shuffled_CM_data(test_data_index,:);

    %% target, prediction calc
    shuffled_index = shuffled_CM_data(:,1);
    ori_time_list = CM_data(:,2);
    CM_data = CM_data(:,3:end);

    % pre-allocate matrices
    prediction_Grad_list = zeros(1+Np, state_num*TEST_NUM); 
    prediction_noGrad_list = zeros(1+Np, state_num*TEST_NUM); 
    prediction_analytic_list = zeros(1+Np, state_num*TEST_NUM);
    obs_state_list = zeros(1+Np, state_num*TEST_NUM); 
    report_list = zeros(TEST_NUM, 5); 
    control_list = zeros(Np, control_num*TEST_NUM);
    
    errrrrrr = zeros(TEST_NUM, state_num);

    % prediction start
    q = 1;
    fprintf("Prediction Start\n")            
%     for sample = shuffled_CM_data
    while true
        if q == TEST_NUM+1
            break
        end
        
        test_idx = shuffled_index(randi(length(shuffled_index)));
        time_step_list = ori_time_list(test_idx+1:test_idx+Np-1, 1) - ori_time_list(test_idx:test_idx+Np-2, 1);

        time_step_test = sum(time_step_list < 0.011) == Np-1;
        
        if time_step_test
%             fprintf("test index: %d\n", test_idx)
            pred_target = CM_data(test_idx, 1:3);
            sample = CM_data(test_idx, 4:end);
            pred_target = pred_target';
            sample = sample';
        else
%             fprintf("non-proper index %d\n", test_idx)
            continue
        end

        sample_num = length(sample);

        % analytic calc
        f0_tmp = analy_F(sample);
        dfdx0_tmp = analy_dFdX(sample);

        % input preparation for dl tookbox
        input_sample = dlarray(sample, "CB");
        
        % NN Uncertainty Prediction Test =======================================
        g0_tmp = extractdata(predict(nn, input_sample));        
        prediction_err = pred_target - f0_tmp - g0_tmp;
        errrrrrr(q, :) = pred_target - f0_tmp;
        report_list(q, 2) = norm(prediction_err, 2);  
    
        % NN Gradient Calc =====================================================
        dgdx0_tmp = zeros(3,sample_num);
        for i = 1:1:3
            [v,grad] = dlfeval(@model,nn,input_sample, i);
            tmp = extractdata(grad); % input_sample = extractdata(input_sample);
            dgdx0_tmp(i,:) = tmp;
        end

        f0 = zeros(sample_num-state_num, 1);
        f0(1:3) = f0_tmp;
        g0 = zeros(sample_num-state_num, 1);
        g0(1:3) = g0_tmp;        
        dgdx0 = zeros(sample_num-state_num, sample_num);
        dgdx0(1:3,1:sample_num) = dgdx0_tmp;        
        dfdx0 = zeros(sample_num-state_num, sample_num);
        dfdx0(1:3,1:6) = dfdx0_tmp;
    
        dfdx0 = dfdx0(:,[1 2 3 7 8 9 10   4 5 6]);
        dgdx0 = dgdx0(:,[1 2 3 7 8 9 10   4 5 6]);        
        sample = sample([1 2 3 7 8 9 10   4 5 6]);


        % current state
        cur_state = sample(1:7);
    
        % sample observed trajectory and control from CM
        obs_traj = CM_data(test_idx:test_idx+Np-1, 1:end);
        obs_state = obs_traj(:,4:6);
        obs_state_list(:, (q-1)*3+1:(q-1)*3+3) = [cur_state(1:3)'; obs_state];
        report_list(q, 1) = obs_state(1) * 3.6;

        obs_control = obs_traj(:,7:9);
        control_list(:, (q-1)*3+1:(q-1)*3+3) = obs_control;
        obs_control = obs_control';
        controlInput = reshape(obs_control, [], 1);
    
        % nominal point
        Dx_analytic = f0 - dfdx0 * sample;        
        Dx_proposed = Dx_analytic + g0 - dgdx0 * sample;
        
        Dx_analytic = Dx_analytic * Ts;  
        Dx_proposed = Dx_proposed * Ts;    

        % prediction calc main
        [traj, err] = pred_err_calc(dfdx0+dgdx0, Dx_proposed, cur_state, obs_state, controlInput, Np, Nc, Ts);
        prediction_Grad_list(:, (q-1)*3+1:(q-1)*3+3) = traj;
        report_list(q, 3) = norm(err, 2);

        [traj, err] = pred_err_calc(dfdx0, Dx_proposed, cur_state, obs_state, controlInput, Np, Nc, Ts);
        prediction_noGrad_list(:, (q-1)*3+1:(q-1)*3+3) = traj;
        report_list(q, 4) = norm(err, 2);

        [traj, err] = pred_err_calc(dfdx0, Dx_analytic, cur_state, obs_state, controlInput, Np, Nc, Ts);
        prediction_analytic_list(:, (q-1)*3+1:(q-1)*3+3) = traj;
        report_list(q, 5) = norm(err, 2);
    
        q = q+1;
    end
    
    %% plot and report
    report_list = array2table(report_list, 'VariableNames', ...
        {'Vx0', 'pred_err', 'proposed', 'noGrad', 'analytic'});
    varfun(@mean, report_list, 'InputVariables', @isnumeric)

    traj_err = [ ...
        sqrt(mean(reshape(mean((prediction_Grad_list - obs_state_list).^2), [], TEST_NUM)'))
        sqrt(mean(reshape(mean((prediction_noGrad_list - obs_state_list).^2), [], TEST_NUM)'))
        sqrt(mean(reshape(mean((prediction_analytic_list - obs_state_list).^2), [], TEST_NUM)'))
]

%     mean(errrrrrr)

    x_axis = 0:1:Np;
    if PLOT_DATA
        for s = 1:1:TEST_NUM
            figure(s)
            tiledlayout(3,1);
    
            nexttile
            plot(x_axis, prediction_Grad_list(:, (s-1)*3+1)*3.6, 'r');
            hold on 
            plot(x_axis, prediction_noGrad_list(:, (s-1)*3+1)*3.6, 'k');        
            plot(x_axis, prediction_analytic_list(:, (s-1)*3+1)*3.6, 'b');
            plot(x_axis, obs_state_list(:, (s-1)*3+1)*3.6, 'g');
            xlabel("Time Step [0.01ms]")
            ylabel("Vx [km/h]") 
            grid on
        
            nexttile
            plot(x_axis, prediction_Grad_list(:, (s-1)*3+2)*3.6, 'r');
            hold on 
            plot(x_axis, prediction_noGrad_list(:, (s-1)*3+2)*3.6, 'k');   
            plot(x_axis, prediction_analytic_list(:, (s-1)*3+2)*3.6, 'b');   
            plot(x_axis, obs_state_list(:, (s-1)*3+2)*3.6, 'g');     
            xlabel("Time Step [0.01ms]")
            ylabel("Vy [km/h]") 
            grid on
        
            nexttile
            plot(x_axis, prediction_Grad_list(:, (s-1)*3+3)*180/pi, 'r');
            hold on
            plot(x_axis, prediction_noGrad_list(:, (s-1)*3+3)*180/pi, 'k');         
            plot(x_axis, prediction_analytic_list(:, (s-1)*3+3)*180/pi, 'b'); 
            plot(x_axis, obs_state_list(:, (s-1)*3+3)*180/pi, 'g');
            xlabel("Time Step [0.01ms]")
            ylabel("YawRate [deg/s]") 
            grid on
    
            lgd = legend('proposed prediction', 'without gradient prediction', 'analytic prediction', 'ground truth');
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
        Fyr = 2 * Ca * (      - ((y_dot-lr*yaw_dot)/ x_dot));
    
        del_Fxf = 0;
        del_Fxr = Frr - Frl;
    
        x_ddot = ((Fxf * cos(delta) - Fyf * sin(delta)) + Frl+Frr) * 1/m + yaw_dot*y_dot -0;
        y_ddot = ((Fxf * sin(delta) + Fyf * cos(delta)) + Fyr) * 1/m - yaw_dot*x_dot;
        psi_ddot = ((lf * (Fxf * sin(delta) + Fyf * cos(delta)) - lr * Fyr) + w * (del_Fxf + del_Fxr)) / Iz;
    
        f = [x_ddot; y_ddot; psi_ddot];
    end
    %% dFdX; analytic gradient
    function dfdx = analy_dFdX(sample)
        Ca = 756.349/(0.6*pi/180);
% Ca = 2802.731/(1.2*pi/180);
        m = 1644.802;
        Iz = 2488.893;
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
        A = h(:,1:7);
        B = h(:,8:end);
        C = eye(3, 7);
        
%         [stateSize, inputSize] = size(C);
        stateSize = 7; inputSize = 3; outputSize = 3;
        predStateSize = outputSize * Np;
        predInputSize = inputSize * Nc;
    
        A = A*Ts+eye(stateSize);
        B = B*Ts;
    
        F = ones(predStateSize, stateSize);
        Phi = zeros(predStateSize, predInputSize);
        gamma = zeros(predStateSize, stateSize);
    
        pre_gamma = zeros(outputSize,stateSize);
        for k = 1:1:Np
            F((k-1)*outputSize+1:k*outputSize, ...
                1:stateSize) = C * A^k;
            gamma((k-1)*outputSize+1:k*outputSize, ...
                1:stateSize) = C * A^(k-1) + pre_gamma;
            pre_gamma = gamma((k-1)*outputSize+1:k*outputSize, 1:stateSize);
            for j = 1:1:Nc
                tmp =  C * A^(k-1) * B;
                Phi((k+j-2)*outputSize+1:(k+j-1)*outputSize, ...
                    (j-1)*inputSize+1:j*inputSize) = tmp;
            end
        end
        
        Phi = Phi(1:predStateSize,1:predInputSize);
    end
end
