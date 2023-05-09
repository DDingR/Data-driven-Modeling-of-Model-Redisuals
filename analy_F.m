%% F; analytic system model
    function f = analy_F(sample)
        Ca = 756.349/(0.6*pi/180);
        m = 1644.80;
        Iz = 2488.892;
        lf = 1.240;
        lr = 1.510;
        w = 0.8;
    
        x_dot = sample(1);
        y_dot = sample(2) ;   
        yaw_dot = sample(3);
        delta = sample(4);
        Frl = sample(5);
        Frr = sample(6);
    
        Fxf = 0;
        Fyf = 2 * Ca * (delta - ((y_dot+lf*yaw_dot)/ x_dot));
        Fyr = 2 * Ca * (      - ((y_dot-lr*yaw_dot)/ x_dot));
    
        del_Fxf = 0;
        del_Fxr = Frr - Frl;
    
        x_ddot = ((Fxf * cos(delta) - Fyf * sin(delta)) + Frl+Frr) * 1/m + yaw_dot*y_dot;
        y_ddot = ((Fxf * sin(delta) + Fyf * cos(delta)) + Fyr) * 1/m - yaw_dot*x_dot;
        psi_ddot = ((lf * (Fxf * sin(delta) + Fyf * cos(delta)) - lr * Fyr) + w * (del_Fxf + del_Fxr)) / Iz;
    
        f = [x_ddot; y_ddot; psi_ddot];
    end