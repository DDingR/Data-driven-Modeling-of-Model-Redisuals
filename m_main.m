% MAIN FUNCTION TO EXECUTE THE MATLAB FUNCTION 
format shortEng
format compact
%% prediction test
NN_NAME_LIST = [ ...
    "0501_1040PM/0"
    "0501_1040PM/30"
    "0501_1040PM/113"
    "0501_1040PM/248"
    "0501_1040PM/FINAL"
    ]';

seed = rng("Shuffle").Seed;
PLOT_DATA = true;

for NN_NAME = NN_NAME_LIST
    rst = prediction_check(PLOT_DATA, seed, NN_NAME);
    
    disp(rst)
end