% MAIN FUNCTION TO EXECUTE THE MATLAB FUNCTION 

%% data pre-processing
PLOT_DATA = false;
file_name = "0421_0641PM0";

trg_data_dir = data_pre_processing(file_name, PLOT_DATA);
%% prediction test
NN_NAME_LIST = [ ...
    "0426_0243PM/79"
    "0427_0702PM/18"
]';

% result = [];
    seed = rng("Shuffle").Seed;

for NN_NAME = NN_NAME_LIST
    prediction_check(seed, NN_NAME)


end