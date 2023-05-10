    %% prediction err calc
    function [traj] = pred_err_calc(f, Dx, cur_state, controlInput, Np, Nc, Ts)
        [Phi, F, gamma] = predmat(f, Np, Nc, Ts);
        traj = F*cur_state + Phi * controlInput + gamma * Dx;
        traj = reshape(traj, 3, []);
        traj = traj';

        traj = [cur_state'; traj];
    end