    %% predicitive matrices; for non-augmented system
    function [Phi, F, gamma] = predmat(h, Np, Nc, Ts)
    % X = F * x_current + Phi * controlSequence + gamma * Dx
    %   Dx means linearization constants
    %
    % these matrices are not for augmented system!!!
        A = h(:,1:3);
        B = h(:,4:6);
        C = eye(3);
        
        [stateSize, inputSize] = size(C);
        predStateSize = stateSize * Np;
        predInputSize = inputSize * Nc;
    
        A = A*Ts+eye(stateSize);
        B = B*Ts;
    
        F = ones(predStateSize, stateSize);
        Phi = zeros(predStateSize, predInputSize);
        gamma = zeros(predStateSize, stateSize);
    
        pre_gamma = zeros(3,3);
        for k = 1:1:Np
            F((k-1)*stateSize+1:k*stateSize, ...
                1:stateSize) = C * A^k;
            gamma((k-1)*stateSize+1:k*stateSize, ...
                1:stateSize) = A^(k-1) + pre_gamma;
            pre_gamma = gamma((k-1)*stateSize+1:k*stateSize, 1:stateSize);
            for j = 1:1:Nc
                tmp =  C * A^(k-1) * B;
                Phi((k+j-2)*stateSize+1:(k+j-1)*stateSize, ...
                    (j-1)*inputSize+1:j*inputSize) = tmp;
            end
        end
        
        Phi = Phi(1:predStateSize,1:predInputSize);
    end