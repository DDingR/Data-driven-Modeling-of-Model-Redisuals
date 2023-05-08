function trg_data_dir = MATLAB_data_pre_processing(file_name, PLOT_DATA)
    %%
    close all
    if nargin < 2
        PLOT_DATA = false;
%         file_name = "0503_0649PM";
        file_name = "0508_1258PM";
    end
    %%
    root_file = "CM_data_collector/results/" + file_name;

    root_info = dir(root_file);
    root_num = length(root_info);

    data = [];

    for j = 3:root_num
%     for j = 
    %      raw_data_dir = "CM_data_collector/results/0501_0133PM";
        raw_data_dir = root_file + "/" +  root_info(j).name;
        

        file_info = dir(raw_data_dir);
        file_num = length(file_info) - 2; % ignore ".", ".." files
    
        for idx = 1:1:file_num
            tmp = readtable(raw_data_dir + "/" + idx + ".csv");
            tmp = table2array(tmp);
            data = [data tmp];
        end
    end
        [var_num, sample_num] = size(data);
    %% DATA INFO
%     var_list = [
%         "Time", 
%         "Car.ax", "Car.ay", "Car.YawAcc",
%         "Car.vx", "Car.vy", "Car.YawRate",
%         # "Car.Aero.Frx_1.x",
%         # "Car.WheelSpd_FL", "Car.WheelSpd_FR", "Car.WheelSpd_RL", "Car.WheelSpd_RR",
%         "Car.FxRL", "Car.FxRR",
%         "Car.FyRL", "Car.FyRR",
%         'Car.SteerAngleFL', 'Car.SteerAngleFR',
%         "Car.FxFL", "Car.FxFR",
%         "Car.FyFL", "Car.FyFR", 
%         'Car.SlipAngleFL', 'Car.SlipAngleFR',
%         'Car.SlipAngleRL', 'Car.SlipAngleRR',
%         ]
    %% for steer
    data(12,:) = (data(12,:) + data(13,:)) * 1/2;
    %% sampling
    sample_list = [1   2 3 4   5 6 7   12 8 9 ];%   18 19 20 21];
    dataset = zeros(sample_num, length(sample_list) + 1);
    for i = 1:1:sample_num
    %     dataset(i,:) = [data(2,i+1) data(sample_list, i)'];
        dataset(i,:) = [i data(sample_list, i)'];
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
        for i = 1:1:var_num
            figure(i)
            plot(data(1,:), data(i,:))
        end
    end
end
