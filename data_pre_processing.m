clear

PLOT_DATA = false;
file_name = "0421_0641PM0";

%%
raw_data_dir = "raw_csv_data/" + file_name + "/" + file_name + ".csv";

data = csvread(raw_data_dir);
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
%         ]
%
%   sample_data = [ax vx vy yawRate FRL FRR StrAng] ->  7
%% sampling
sample_list = [5 6 7 12 8 9];
dataset = zeros(sample_num, 10);
for i = 1:1:sample_num
%     dataset(i,:) = [data(2,i+1) data(sample_list, i)'];
    dataset(i,:) = [i data(2:4,i)' data(sample_list, i)'];
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

