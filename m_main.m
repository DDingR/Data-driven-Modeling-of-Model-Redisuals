% MAIN FUNCTION TO EXECUTE THE MATLAB FUNCTION 
format shortEng
format compact
%% prediction test
NN_NAME_LIST = [ ...
    "0426_0243PM/79"
%     "0427_0702PM/18"
    "0427_0803PM/FINAL"
]';

seed = rng("Shuffle").Seed;
PLOT_DATA = true;

for NN_NAME = NN_NAME_LIST
    prediction_check(PLOT_DATA, seed, NN_NAME)


end