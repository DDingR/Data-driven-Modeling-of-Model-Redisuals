layers = [
    imageInputLayer([32 1],'Name','input','Normalization','none')
    fullyConnectedLayer(Nhide,"Name","FC1")
    reluLayer("Name","Relu1")
    fullyConnectedLayer(Nhide,"Name","FC2")
    dropoutLayer(0.5,"Name","DO")
    fullyConnectedLayer(outputSize,"Name","FC3")
    reluLayer('Name','Relu2')];
lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);
% Training Option
numEpochs = 1e3;
miniBatchSize = 32;
initialLearnRate = 0.001;
decay = 0.01;
momentum = 0.9;
plots = "training-progress";
executionEnvironment = "auto";
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end
%% Training The Network
numObservations = numel(output);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
iteration = 0;
start = tic;
% Loop over epochs.
for epoch = 1:numEpochs
    % Shuffle data.
    idx = randperm(numel(OutPower(:,1)));
    input = input(:,:,:,idx);
    output = output(:,idx);
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = input(:,:,:,idx);
        Y1 = output(1,idx);
        Y2 = output(2,idx);
        Y3 = output(3,idx);
        Y4 = output(4,idx);
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(X,'SSCB');
        dlY1= dlarray(Y1,'SB');
        dlY2= dlarray(Y2,'SB');
        dlY3= dlarray(Y3,'SB');
        dlY4= dlarray(Y4,'SB');
        %         dlY = dlarray(Y,'SSCB');
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX,dlY1,dlY2,dlY3,dlY4);
        dlnet.State = state;


    end
end

        function [gradients,state,loss] = modelGradients(dlnet,dlX,Y1,Y2,Y3,Y4)
        [dlYPred,state] = forward(dlnet,dlX);
        loss = sqrt((dlYPred(1)-Y1).^2+(dlYPred(2)-Y2).^2+(dlYPred(3)-Y3).^2+(dlYPred(4)-Y4).^2)/2;
        gradients = dlgradient(loss,dlnet.Learnables);
        end