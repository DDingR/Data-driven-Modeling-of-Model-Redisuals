clear

nn = "./savemodel/0424_0544PMNN_21.onnx";
nn = importONNXNetwork( ...
  nn,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
);

x = dlarray([1.0,1.0,1.0,1.0,1.0,1.0], "BC");

[f,g] = dlfeval(@model,nn,x)


function [y, g] = model(net, x)
   y = forward(net, x);
   % g = dlgradient(y, net.Learnables);
   g = dlgradient(y, x);
end

