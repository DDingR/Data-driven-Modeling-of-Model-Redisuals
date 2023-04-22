clear
clf(1); clf(2); clf(3); clf(4)
%% test range
x1 = 1.5:0.01:4.5;
x2 = 1.5:0.01:4.5;

[X1,X2] = meshgrid(x1,x2);

%% NN import
nn = "./savemodel/NN_FINAL.onnx";
nn = importONNXNetwork( ...
  nn,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
);
% analyzeNetwork(nn)

nn_non_train = "./savemodel/NN_EPISODE0.onnx";
nn_non_train = importONNXNetwork( ...
  nn_non_train,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
);
% analyzeNetwork(nn)


%% target, prediction calc
trg =zeros(length(x1), length(x2)); 
NN = zeros(length(x1), length(x2));
NN_non_train = zeros(length(x1), length(x2));

for i = 1:1:length(x1)
    for j = 1:1:length(x2)
        trg(i,j) = 1/3*sin(x1(i)/3+pi) * 1/2*sin(x2(j)) + 1/5*sin(1/3)*x2(j);
        NN(i,j) = predict(nn, dlarray([x1(i) x2(j) 1], "BC"));
        NN_non_train(i,j) = predict(nn_non_train, dlarray([x1(i) x2(j) 1], "BC"));
    end
end
%% plot
% target plot
figure(1)
mesh(X1,X2,trg)
view(0,90)
title("Target")
xlabel("X1") 
ylabel("X2")
colorbar

% NN predict plot
figure(2)
mesh(X1,X2,NN)
view(0,90)
title("Predict")
xlabel("X1") 
ylabel("X2")
colorbar

% error plot
figure(3)
mesh(X1,X2,abs(trg-NN))
view(0,90)
title("Abs Error")
xlabel("X1") 
ylabel("X2")
colorbar

% not trained NN predict plot
figure(4)
mesh(X1,X2,NN_non_train)
view(0,90)
title("Not Trained Predict")
xlabel("X1") 
ylabel("X2")
colorbar
%% original function
% def G(X):
%     x1, x2, u = X
%     return 1/3*np.sin(x1/3+np.pi)*1/2*np.sin(x2) + 1/5*np.sin(u/3)*x2
% 
% def dG(X):
%     x1, x2, u = X
%     return [
%         1/3*1/3*np.cos(x1/3+np.pi)*1/2*np.sin(x2),
%         1/3*np.sin(x1/3+np.pi)*1/2*np.cos(x2) + 1/5*np.sin(u/3),
%         1/5*1/3*np.cos(u/3)*x2