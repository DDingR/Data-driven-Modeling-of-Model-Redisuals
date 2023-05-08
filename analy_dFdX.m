
        %% dFdX; analytic gradient
    function dfdx = analy_dFdX(sample)
        Ca = 756.349/(0.6*pi/180);
% Ca = 2802.731/(1.2*pi/180);
        m = 1644.802;
        Iz = 2488.893;
        lf = 1.240;
        lr = 1.510;
        w = 0.8;
        
        x_dot = sample(1);
        y_dot = sample(2) ;   
        yaw_dot = sample(3);
        delta = sample(4);        
        % Frl = sample(:,4);
        % Frr = sample(:,5);
        
        dfdx_op = [
            [                                               -(2*Ca*sin(delta)*(y_dot + lf*yaw_dot))/(m*x_dot^2),             yaw_dot + (2*Ca*sin(delta))/(m*x_dot),                   y_dot + (2*Ca*lf*sin(delta))/(m*x_dot)]
            [((2*Ca*(y_dot - lr*yaw_dot))/x_dot^2 + (2*Ca*cos(delta)*(y_dot + lf*yaw_dot))/x_dot^2)/m - yaw_dot,       -((2*Ca)/x_dot + (2*Ca*cos(delta))/x_dot)/m, ((2*Ca*lr)/x_dot - (2*Ca*lf*cos(delta))/x_dot)/m - x_dot]
            [  -((2*Ca*lr*(y_dot - lr*yaw_dot))/x_dot^2 - (2*Ca*lf*cos(delta)*(y_dot + lf*yaw_dot))/x_dot^2)/Iz, ((2*Ca*lr)/x_dot - (2*Ca*lf*cos(delta))/x_dot)/Iz,   -((2*Ca*cos(delta)*lf^2)/x_dot + (2*Ca*lr^2)/x_dot)/Iz]
        ];
        
        dfdu_op = [ 
            [      -(2*Ca*sin(delta) + 2*Ca*cos(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/m,   1/m,  1/m]
            [       (2*Ca*cos(delta) - 2*Ca*sin(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/m,     0,    0]
            [(2*Ca*lf*cos(delta) - 2*Ca*lf*sin(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/Iz, -w/Iz, w/Iz]
        ];
        
        dfdx = [dfdx_op dfdu_op];
    end