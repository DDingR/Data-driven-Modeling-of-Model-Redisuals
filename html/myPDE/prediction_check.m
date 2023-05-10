function [] = prediction_check(PLOT_DATA, NN_NAME, TEST_NUM, Ts, Np)
    close all
    fprintf("\n======== Prediction Test at %s ========\n", char(datetime))
    %% args
    if nargin == 3
        TEST_NUM = 5;
        Ts = 0.01; Np = 100; Nc = Np;
    elseif nargin < 5
        NN_NAME = "FINAL10000"; % 0507_0746PM

        TEST_NUM = 5;
        Ts = 0.01; Np = 100; Nc = Np;
        
        PLOT_DATA = true;
%         PLOT_DATA = false;
    end

    %% overwrite constant!

    %% simulation constants
    state_num = 3;
    control_num = 3;
    %% prediction test constants
    format shortEng
    format compact
    %% network load
    fprintf("Loading Neural Network NN_NAME: %s\n", NN_NAME)    
    nn = "./" + NN_NAME + ".onnx";
    nn = importONNXNetwork( ...
      nn,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
    );
    % analyzeNetwork(nn)    

    %% target, prediction calc

    % pre-allocate matrices
    prediction_Grad_list = zeros(1+Np, state_num*TEST_NUM); 
    prediction_analytic_list = zeros(1+Np, state_num*TEST_NUM);

    % prediction start
    q = 1;
    fprintf("Prediction Start\n")            
    while true
        if q == TEST_NUM+1
            break
        end

        sample = (rand([6,1]) - 1/2) * 2 * 1.5;
    controlInput = zeros(Np, control_num);        
        controlInput(2:end,:) = (rand([Np-1,3]) - 1/2) * 2 * 1.5;
        controlInput(1,:) = sample(4:6)';
        controlInput = controlInput';
        controlInput = reshape(controlInput, [], 1);

        g0 = analy_G(sample);
        dgdx0 = analy_dGdX(sample);
    
        % input preparation for dl tookbox
        input_sample = dlarray(sample, "CB");

        % NN Uncertainty Prediction Test =======================================
        pred_g0 = extractdata(predict(nn, input_sample));        
        prediction_err = g0 - pred_g0 
    
        % NN Gradient Calc =====================================================
        pred_dgdx0 = zeros(3,6);
        for i = 1:1:3
            [v,grad] = dlfeval(@model,nn,input_sample, i);
            tmp = extractdata(grad); % input_sample = extractdata(input_sample);
            pred_dgdx0(i,:) = tmp(1:6);
        end
        
        % current state
        cur_state = sample(1:3);
    
        % nominal point
        Dx = g0 - dgdx0 * sample;
        Dx_pred = pred_g0 - pred_dgdx0 * sample;
        
        Dx = Dx * Ts;  
        Dx_pred = Dx_pred * Ts;    

        % prediction calc main
        [traj] = pred_err_calc(pred_dgdx0, Dx_pred, cur_state, controlInput, Np, Nc, Ts);
        prediction_Grad_list(:, (q-1)*3+1:(q-1)*3+3) = traj;

        [traj] = pred_err_calc(dgdx0, Dx, cur_state, controlInput, Np, Nc, Ts);
        prediction_analytic_list(:, (q-1)*3+1:(q-1)*3+3) = traj;
    
        q = q+1;
    end
    
    %% plot and report
    x_axis = 0:1:Np;
    if PLOT_DATA
        if TEST_NUM <= 5
            PLOT_NUM = TEST_NUM;
        else
            PLOT_NUM = 5;
        end
            figure(PLOT_NUM+1)
            tiledlayout(3,1);

        for s = 1:1:PLOT_NUM
            figure(s)
            tiledlayout(3,1);
    
            nexttile
            plot(x_axis, prediction_Grad_list(:, (s-1)*3+1), 'r');
            hold on 
            plot(x_axis, prediction_analytic_list(:, (s-1)*3+1), 'b');
            xlabel("Time Step [0.01ms]",'fontsize',10,'fontname', 'Times New Roman')
            ylabel("Vx [m/s]",'fontsize',10,'fontname', 'Times New Roman') 
            grid on
        
            nexttile
            plot(x_axis, prediction_Grad_list(:, (s-1)*3+2), 'r');
            hold on 
            plot(x_axis, prediction_analytic_list(:, (s-1)*3+2), 'b');   
            xlabel("Time Step [0.01ms]",'fontsize',10,'fontname', 'Times New Roman')
            ylabel("Vy [m/s]",'fontsize',10,'fontname', 'Times New Roman')
            grid on
        
            nexttile
            plot(x_axis, prediction_Grad_list(:, (s-1)*3+3), 'r');
            hold on
            plot(x_axis, prediction_analytic_list(:, (s-1)*3+3), 'b'); 
            xlabel("Time Step [0.01ms]",'fontsize',10,'fontname', 'Times New Roman')
            ylabel("YawRate [rad/s]",'fontsize',10,'fontname', 'Times New Roman') 
            grid on    
        end
    end    


    %% function to calculate gradient
    function [y, g] = model(net, x, i)
       y = forward(net, x);
       y = y(i);
       g = dlgradient(y, x);
    end    
end
   