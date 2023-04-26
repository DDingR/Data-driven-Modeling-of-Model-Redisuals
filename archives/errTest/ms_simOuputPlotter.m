clear

% ergName = '~/CM_Projects/demo1/SimOutput/demo1/DDingUbuntu/20230410/ms_road_friction_191132_case0.erg';
ergName = 'ms_road_friction_191132_case0.erg';

data = cmread(ergName);

%% pre-processing

t = data.Time.data;


%% state
figure(1)
% hold on
grid on

subplot(3, 1, 1)
plot(t, data.Car_vx.data)
subplot(3, 1, 2)
plot(t, data.Car_vy.data)
subplot(3, 1, 3)
plot(t, data.Car_YawRate.data)

%% control
figure(2)
grid on

subplot(3, 1, 1)
plot(t, data.Vhcl_Steer_Ang.data)
subplot(3, 1, 2)
plot(t, data.Vhcl_RL_Fx.data)
subplot(3, 1, 3)
plot(t, data.Vhcl_RR_Fx.data)

