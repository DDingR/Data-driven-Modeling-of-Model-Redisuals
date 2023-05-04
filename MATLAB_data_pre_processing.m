function trg_data_dir = data_pre_processing(file_name, PLOT_DATA)
    %%
    if nargin < 2
        PLOT_DATA = false;
%         file_name = "0503_0649PM";
        file_name = "0503_0649PM";
    end
    %%
    raw_data_dir = "CM_data_collector/results/" + file_name;
%      raw_data_dir = "CM_data_collector/results/0501_0133PM";

    file_info = dir(raw_data_dir);
    file_num = length(file_info) - 2; % ignore ".", ".." files

    data = [];
    for idx = 1:1:file_num
        tmp = readtable(raw_data_dir + "/" + idx + ".csv");
        tmp = table2array(tmp);
        data = [data tmp];
    end

    [var_num, sample_num] = size(data);
    %% DATA INFO
    %     var_list = [
    %         "Time", 
    %         "Car.ax", "Car.ay", "Car.YawAcc",
    %         "Car.vx", "Car.vy", "Car.YawRate",
    %         # "Car.Aero.Frx_1.x",
    %         # "Car.WheelSpd_FL", "Car.WheelSpd_FR", "Car.WheelSpd_RL", "Car.WheelSpd_RR",
    %         # "Car.FxFL", "Car.FxFR", 
    %         "Car.FxRL", "Car.FxRR",
    %         # "Car.FyFL", "Car.FyFR", 
    %         "Car.FyRL", "Car.FyRR",
    %         "VC.Steer.Ang"
%         'Car.SteerAngleFL', 'Car.SteerAngleFR'
    
    %         ]
    %
    %   sample_data = [ax vx vy yawRate FRL FRR StrAng] ->  7
    %% for steer
    data(12,:) = (data(12,:) + data(13,:)) * 1/2;
    %% sampling
    sample_list = [5 6 7 12 8 9];
    dataset = zeros(sample_num, 11);
    for i = 1:1:sample_num
    %     dataset(i,:) = [data(2,i+1) data(sample_list, i)'];
        dataset(i,:) = [i data(1:4,i)' data(sample_list, i)'];
    end
    trg_data = array2table(dataset);
    trg_data_dir = "processed_csv_data/" + file_name + ".csv";
    writetable(trg_data, trg_data_dir, WriteVariableNames=false);
    disp("saved at " + trg_data_dir)
    
    shuffled_dataset = dataset(randperm(sample_num),:);
    %% save
    trg_data = array2table(shuffled_dataset);
    trg_data_dir = "processed_csv_data/shuffled_" + file_name + ".csv";
    writetable(trg_data, trg_data_dir, WriteVariableNames=false);
    disp("saved at " + trg_data_dir)
    
    %% plotter
    if PLOT_DATA
        for i = 2:1:var_num
            figure(i)
            plot(data(1,:), data(i,:))
        end
    end
end
