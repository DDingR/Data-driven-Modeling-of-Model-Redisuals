%% F; analytic system model
    function f = analy_G(sample)
    
        x1 = sample(1);
        x2 = sample(2) ;   
        x3 = sample(3);
        u1 = sample(4);
        u2 = sample(5);
        u3 = sample(6);
    
        x1_ddot = ((cos(u1) - (x2+x3) * sin(u1)) + u2+u3) + x3*x2;
        x2_ddot = ((sin(u1) + (x2+x3) * cos(u1)) + (x2-x3)) - x3*x1;
        x3_ddot = (((sin(u1) + (x2+x3) * cos(u1)) - (x2-x3)) + u3-u2);
    
        f = [x1_ddot; x2_ddot; x3_ddot];
    end