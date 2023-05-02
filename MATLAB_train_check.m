% MAIN FUNCTION TO EXECUTE THE MATLAB FUNCTION 
format shortEng
format compact
%% prediction test
% NN_NAME_LIST = [ ...
%     "0501_1040PM/0"
%     "0501_1040PM/30"
%     "0501_1040PM/113"
%     "0501_1040PM/248"
%     "0501_1040PM/FINAL"
%     ]';

NN_dir = "0501_1040PM";
NN_info = dir("savemodel/" + NN_dir);
NN_NAME_LIST = [];
for l = 1:1:NN_num
    NN_NAME_LIST = [NN_NAME_LIST NN_dir + "/" +  NN_info(2+l).name(1:end-5)];
end

FILE_NAME = "0501_0133PM";

seed = rng("Shuffle").Seed;
PLOT_DATA = false;
TRAIN_CHECK = true;
TEST_NUM = 5;
Ts = 0.01; Np = 20;

rst_list = []; j = 1;
for NN_NAME = NN_NAME_LIST
    rst = prediction_check(PLOT_DATA, seed, NN_NAME, FILE_NAME, TEST_NUM, Ts, Np)
    rst = table2array(rst);

    if isempty(rst_list)
        rst_list = rst;
    else
        rst_list(:,:,j) = rst;
    end
    j = j + 1;
end

if TRAIN_CHECK
    for k = 1:1:TEST_NUM
        figure(k+TEST_NUM)

        tmp = rst_list(k,3,:);
        tmp = reshape(tmp, [1, length(NN_NAME_LIST)]);
        anal = rst_list(k,5,:);
        anal = reshape(anal, [1, length(NN_NAME_LIST)]);

        plot(tmp)
        hold on
        plot(anal)
    end
end