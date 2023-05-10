    %% prediction err calc
    function [traj, err] = pred_err_calc(f, Dx, cur_state, obs_state, controlInput, Np, Nc, Ts)
        [Phi, F, gamma] = predmat(f, Np, Nc, Ts);
        traj = F*cur_state + Phi * controlInput + gamma * Dx;
        traj = reshape(traj, 3, []);
        traj = traj';
        
        if obs_state == 0
            err = 0;
        else
            err = (obs_state(2:end,:) - traj).^2;
            err = mean(err);    
        end
        traj = [cur_state'; traj];
    end