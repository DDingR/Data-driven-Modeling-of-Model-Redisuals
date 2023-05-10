% MAIN FUNCTION TO EXECUTE THE MATLAB FUNCTION 
format shortEng
format compact
%% prediction test
PLOT_DATA = false;
TRAIN_CHECK = true;
TEST_NUM = 10;
Ts = 0.01; Np = 20;

NN_dir = "0510_0238PM_MY_PDE";
NN_full_dir = "0510_0235PM_MY_PDE";
%%
NN_info = dir(NN_dir);
NN_num = length(NN_info)-2;
NUM_LIST = [];
NN_NAME_LIST = [];
for l = 1:1:NN_num
    num = NN_info(2+l).name(1:end-5);
    num = str2double(num);
    if rem(num, 1000) == 0 || isnan(num)
        NUM_LIST = [NUM_LIST num];
    end
end
NUM_LIST = sort(NUM_LIST);

for l = 1:1:length(NUM_LIST)
    tmp = num2str(NUM_LIST(l));
    if strcmp(tmp, "NaN")
        NN_NAME_LIST = [NN_NAME_LIST NN_dir + "/FINAL10000"];
    else
        NN_NAME_LIST = [NN_NAME_LIST NN_dir + "/" + tmp];
    end
end

NN_info = dir(NN_full_dir);
NN_num = length(NN_info)-2;
NUM_LIST = [];
NN_NAME_FULL_LIST = [];
for l = 1:1:NN_num
    num = NN_info(2+l).name(1:end-5);
    num = str2double(num);
    if rem(num, 1000) == 0 || isnan(num)
        NUM_LIST = [NUM_LIST num];
    end
end
NUM_LIST = sort(NUM_LIST);

for l = 1:1:length(NUM_LIST)
    tmp = num2str(NUM_LIST(l));
    if strcmp(tmp, "NaN")
        NN_NAME_FULL_LIST = [NN_NAME_FULL_LIST NN_full_dir + "/FINAL10000"];
    else
        NN_NAME_FULL_LIST = [NN_NAME_FULL_LIST NN_full_dir + "/" + tmp];
    end
end

%%
rst_list = []; j = 1;

if length(NN_NAME_LIST) ~= length(NN_NAME_FULL_LIST)
    error("not same length of NN and NN_FULL")
end

for k = 1:1:length(NUM_LIST)
    NN_NAME = NN_NAME_LIST(k);
    NN_FULL = NN_NAME_FULL_LIST(k);
    rst = prediction_check(PLOT_DATA, NN_NAME, NN_FULL, TEST_NUM, Ts, Np)
    rst = table2array(rst);
    
    if isempty(rst_list)
        rst_list = rst;
    else
        rst_list(:,:,j) = rst;
    end
    j = j + 1;
 end


rst_list = sqrt(mean(rst_list.^2));

if TRAIN_CHECK
    plot(reshape(rst_list(:,1,:), [], 1), 'r')
    hold on
    plot(reshape(rst_list(:,2,:), [], 1) ,'b')
end