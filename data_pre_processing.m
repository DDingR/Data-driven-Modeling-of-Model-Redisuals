function trg_data = data_pre_processing(file_name, plot_data)

    if file_name == ""
        file_name = "0421_0641PM0";
    end

    raw_data_dir = "raw_csv_data/" + file_name + "/" + file_name + ".csv";

    data = csvread(raw_data_dir);
    [data_num, sample_num] = size(data);

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
    sample_list = [5 6 7 8 9 12];
    dataset = zeros(sample_num, 7);
    for i = 1:1:sample_num
    %     dataset(i,:) = [data(2,i+1) data(sample_list, i)'];
        dataset(i,:) = [data(2,i) data(sample_list, i)'];
    end

    %% save
    trg_data = array2table(dataset);

    trg_data_dir = "processed_csv_data/" + "/" + file_name + ".csv";
    writetable(trg_data, file_name + ".csv", WriteVariableNames=false);


    %% plotter
    if plot_data
        for i = 2:1:data_num
            figure(i)
            plot(data(1,:), data(i,:))
        end
    end

