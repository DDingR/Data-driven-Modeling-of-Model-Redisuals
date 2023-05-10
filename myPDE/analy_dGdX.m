
        %% dFdX; analytic gradient
    function dfdx = analy_dGdX(sample)

       x1 = sample(1);
        x2 = sample(2) ;   
        x3 = sample(3);
        u1 = sample(4);
        u2 = sample(5);
        u3 = sample(6);

 dfdx =  [...      
[  0, x3 - sin(u1),     x2 - sin(u1), - sin(u1) - cos(u1)*(x2 + x3),  1, 1]
[-x3,  cos(u1) + 1, cos(u1) - x1 - 1,   cos(u1) - sin(u1)*(x2 + x3),  0, 0]
[  0,  cos(u1) - 1,      cos(u1) + 1,   cos(u1) - sin(u1)*(x2 + x3), -1, 1]
 ];
    end