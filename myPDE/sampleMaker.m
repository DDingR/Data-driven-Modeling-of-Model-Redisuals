

%% sampling
%% save
trg_data = array2table(shuffled_dataset);
trg_data_dir = "processed_csv_data/shuffled_" + file_name + ".csv";
writetable(trg_data, trg_data_dir, WriteVariableNames=false);
disp("saved at " + trg_data_dir)

