% case 0 no input
% case 1 str ang 0.5 rad
% case 2 RL 100 N
% case 3 RR 100 N


clear;
clf(1);clf(2);clf(3)

interested_steps = 500;

caseNum = 3

delta = 0;
Frxl = 0;
Frxr = 0;
ergName = 'case0.erg';
if caseNum == 1
    delta = 0.5*pi/180;
ergName = 'case1.erg';
elseif caseNum == 2
    Frxl = 100;
    ergName = 'case2.erg';
elseif caseNum == 3
    Frxr = 100;
    ergName = 'case3.erg';
end

% ergName = '~/CM_Projects/demo1/SimOutput/demo1/DDingUbuntu/20230410/ms_road_friction_191132_case0.erg';


data = cmread(ergName);
%% pre-processing

t = data.Time.data;
additional_save_name = "minus 300 mass ";

Ts = 0.01;
%     Cf = 435.418/0.296296;
    Cr = 756.349/(0.6*pi/180);
Cf = Cr;
    m = 1644.80 - 300;
    Iz = 2488.892;
    lf = 1.240;
    lr = 1.510;
    w = 0.8;
    

Np = 5 / 0.01;
Nc = 1;

Vx = 50/3.6;
Vy = 0;
YawRate = 0;

state = [Vx Vy YawRate]';
delState = zeros(3,1);

control = [delta Frxl Frxr]';

controlInput = repmat(control, Nc, 1);

dfdx_op = [

[                                               -(2*Cf*sin(delta)*(Vy + lf*YawRate))/(m*Vx^2),             YawRate + (2*Cf*sin(delta))/(m*Vx),                   Vy + (2*Cf*lf*sin(delta))/(m*Vx)]
[((2*Cr*(Vy - lr*YawRate))/Vx^2 + (2*Cf*cos(delta)*(Vy + lf*YawRate))/Vx^2)/m - YawRate,       -((2*Cr)/Vx + (2*Cf*cos(delta))/Vx)/m, ((2*Cr*lr)/Vx - (2*Cf*lf*cos(delta))/Vx)/m - Vx]
[  -((2*Cr*lr*(Vy - lr*YawRate))/Vx^2 - (2*Cf*lf*cos(delta)*(Vy + lf*YawRate))/Vx^2)/Iz, ((2*Cr*lr)/Vx - (2*Cf*lf*cos(delta))/Vx)/Iz,   -((2*Cf*cos(delta)*lf^2)/Vx + (2*Cr*lr^2)/Vx)/Iz]

];

dfdu_op = [

[      -(2*Cf*sin(delta) + 2*Cf*cos(delta)*(delta - (Vy + lf*YawRate)/Vx))/m,   1/m,  1/m]
[       (2*Cf*cos(delta) - 2*Cf*sin(delta)*(delta - (Vy + lf*YawRate)/Vx))/m,     0,    0]
[(2*Cf*lf*cos(delta) - 2*Cf*lf*sin(delta)*(delta - (Vy + lf*YawRate)/Vx))/Iz, -w/Iz, w/Iz]

];

A = dfdx_op;
B = dfdu_op;
C = eye(size(A));
D = zeros(size(B));

[stateSize, inputSize] = size(C);
augStateSize = stateSize * 2;
predStateSize = stateSize * Np;
predInputSize = inputSize * Nc;

augState = [delState; state];

% Start ========================================================================================
% [A,B,C,D] = augmentation_ext(A,B,C,D,Ts);
% ==============================================================================================
A = A*Ts+eye(stateSize);
B = B*Ts;

A = [A zeros(inputSize, stateSize);
    C*A eye(stateSize)];
B = [B; C*B];
C = [zeros(stateSize, inputSize) eye(stateSize)];
% End ==========================================================================================
% [A,B,C,D] = augmentation_ext(A,B,C,D,Ts);
% ==============================================================================================

% Start ========================================================================================
% [F, Phi] = predMat(A, B, C, Np, Nc);
% ==============================================================================================
F = ones(predStateSize, augStateSize);
Phi = zeros(predStateSize, predInputSize);

for i = 1:1:Np
    F((i-1)*stateSize+1:i*stateSize, ...
        1:augStateSize) = C * A^i;
    for j = 1:1:Nc
        tmp = C * A^(i-1) * B;
        Phi((i+j-2)*stateSize+1:(i+j-1)*stateSize, ...
            (j-1)*inputSize+1:j*inputSize) = tmp;
    end
end

Phi = Phi(1:predStateSize,1:predInputSize);


traj = F*augState + Phi * controlInput;
traj = reshape(traj, 3, []);
%%
mydata = [
    data.Car_vx.data
    data.Car_vy.data
    data.Car_YawRate.data
    data.Vhcl_Steer_Ang.data
    data.Vhcl_RL_Fx.data
    data.Vhcl_RR_Fx.data
    data.Car_ax.data
    data.Car_ay.data
    data.Car_YawAcc.data
    ];

t = t(:,1:interested_steps);
traj = traj(:, 1:interested_steps);
mydata = mydata(:, 1:interested_steps);

f1 = figure(1);
subplot(3,1,1)
title("Vx")
hold on; grid on
plot(t, traj(1,:))
plot(t, mydata(1,:))
subplot(3,1,2)
title("Vy")
hold on; grid on
plot(t, traj(2,:))
plot(t, mydata(2,:))
subplot(3,1,3)
title("YawRate")
hold on; grid on
plot(t, traj(3,:))
plot(t, mydata(3,:))

f2 = figure(2);
subplot(3, 1, 1)
title("delta")
hold on; grid on
plot(t,ones(size(traj(1,:))) * control(1))
plot(t, mydata(4,:))
subplot(3, 1, 2)
title("Frxl")
hold on; grid on
plot(t,ones(size(traj(1,:))) * control(2))
plot(t, mydata(5,:))
subplot(3, 1, 3)
title("Frxr")
hold on; grid on
plot(t,ones(size(traj(1,:))) * control(3))
plot(t, mydata(6,:))

f3 = figure(3);
subplot(3,1,1)
title("ax")
hold on; grid on
plot(t, mydata(7,:))
subplot(3,1,2)
title("ay")
hold on; grid on
plot(t, mydata(8,:))
subplot(3,1,3)
title("YawAcc")
hold on; grid on
plot(t, mydata(9,:))

saveas(f1, additional_save_name + "state case" + string(caseNum) + ".png")
saveas(f2, additional_save_name + "input case" + string(caseNum) + ".png")
saveas(f3, additional_save_name + "accel case" + string(caseNum) + ".png")
