clear all;
close all;
clc;
#number of UAVs
U = 9;
#number of users
K = 12;
#generate the small scale channels between UAVs and users, UAVs and Macro base station (MBS)
generate_channel_rician(U,K);
#load('parameter_9_12.mat');
%N0 = 1e-10;
#generate users positions
generate_ue(K);
%load('parameter_ue_location_12.mat');
#heights of a UAV
H = 150;
#height of MBS
Hb = 30; 
#communication range of a UAV
Range = 200;
eta = 5;
sigma02=1;
% Power budget %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PmbsdBm = 53;
Pmbs = 10^((PmbsdBm - 30)/10);
PUAVdBm = [30];
PUAV = 10.^((PUAVdBm - 30)/10);
% gamma_min =10^0.4;
% Rmin = log(1+gamma_min);
%generate UAV position
coor_uav=zeros(U,2);
for i=1:U
    coor_uav(i,1)=200*rand(1);
    coor_uav(i,2) = 200*rand(1);
end
#distance between UAVs and users
delta=zeros(U,K);
for i=1:U
    for k=1:K
        delta(i,k)=((coor_uav(i,1)-coor_ue(k,1))^2+(coor_uav(i,2)-coor_ue(k,2))^2+H^2)^(-1/2);
    end
end
#distance between UAVs and MBS
deltab=zeros(U,1);
for i=1:U
    deltab(i)=((0-coor_uav(i,1))^2+(0-coor_uav(i,2))^2+(H-Hb)^2)^(-1/2);
end

%pathloss_mbs_uav(coor_uav,coor_ue);
%load('channel_12_8.mat');
%%%%%%%%%%%%%%%%%%%%%%%% Initial points
xi = 1e-2*ones(K,1);%1e-2
zeta = 1e-1*ones(K,1);%1e-1
psi = 1e0*ones(U,1);%1e0
const_delta = 1e-3*ones(U,K);%1e-3
const_deltab = 1e-3*ones(U,1);%1e-3
const_c = 0.1*ones(U,K);%0.1
const_v = 1e-1*ones(U,K);%1e-1
const_p = 2*1e0*ones(U,1);%2e0
%s_const = 1e0*ones(K,1);
A = 200;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nIterMax =30;
testing = zeros(nIterMax,1);
opt_valu = zeros(nIterMax,1);
runtime = zeros(nIterMax,1);
sum_rate = zeros(nIterMax,1);
ops = sdpsettings('solver','mosek','verbose',0);
testing = zeros(nIterMax,1);
opt_valu = zeros(nIterMax,1);
for nIter = 1:nIterMax
    % VARIABLES
    
    delta = sdpvar(U,K,'full','real');
    deltab = sdpvar(U,1,'full','real');
    coor_uav = sdpvar(U,2,'full','real');
    p = sdpvar(U,1,'full','real'); %mbs power to all UAVs
    v = sdpvar(U,K,'full','complex'); %UAVs power to all UEs
    lambda = sdpvar(U,K,'full','real'); % soft power
    mu = sdpvar(K,1,'full','real'); %slack variable for user rate
    c = sdpvar(U,K,'full','real'); %association binary variable relaxed
    z = sdpvar(U,K,'full','real'); %slack variable for c
    
    %s = sdpvar(K,1,'full','real'); %slack variable for SINR
    
    % Objective function
    obj = -rate_lower_ue(v,delta,const_v,const_delta,H_channel,N0,K,xi)+support_func(v,xi,delta,K)+A*sum(sum(z.^2));
    %obj = -sum(mu)+ A*sum(sum(z.^2));
    %constraint
    constraints = [];
    constraints = [p >= 0, 1 >= c >= 0, lambda >= 0, z >= 0];
    
    constraints = [constraints, 250 >= coor_uav];
    constraints = [constraints, -250 <= coor_uav];
    constraints = [constraints, 1e-3 <= delta ];
    constraints = [constraints, 1e-1 >= delta ];
    constraints = [constraints, 1e-3 <= deltab ];
    constraints = [constraints, 1e-1 >= deltab ];
%         
        constraints =[constraints,sum(p)<=Pmbs];
    for iUAV=1:U
        
        constraints =[constraints, sum(lambda(iUAV,:))<=PUAV];
        
        constraints = [constraints,cone([coor_uav(iUAV,1),coor_uav(iUAV,2),(H-Hb)],2/const_deltab(iUAV,1)-1/(const_deltab(iUAV,1))^2*deltab(iUAV,1))];
        constraints = [constraints, rate_upper_ue(v,delta,const_v,const_delta,H_channel,N0,K,zeta,c,const_c,iUAV)+support_func_c(v,zeta,delta,K,c,iUAV) <= rate_lower_uav(p,const_p,deltab,const_deltab,psi,iUAV,G_channel,N0)-support_func_uav(psi,p,deltab,iUAV)];
        
        for iUE=1:K
            constraints =[constraints, 0 <= c(iUAV,iUE) <= 1];
            constraints = [constraints,cone([coor_uav(iUAV,1)-coor_ue(iUE,1),coor_uav(iUAV,2)-coor_ue(iUE,2),H],2/const_delta(iUAV,iUE)-1/(const_delta(iUAV,iUE))^2*delta(iUAV,iUE))];
            constraints = [constraints, cone([coor_uav(iUAV,1)-coor_ue(iUE,1),coor_uav(iUAV,2)-coor_ue(iUE,2)],Range+eta*(1-c(iUAV,iUE)))];
            constraints =[constraints, cone([v(iUAV,iUE),(c(iUAV,iUE)-lambda(iUAV,iUE))/2],(c(iUAV,iUE)+lambda(iUAV,iUE))/2)];
            constraints = [constraints, c(iUAV,iUE)-const_c(iUAV,iUE)^2-2*const_c(iUAV,iUE)*(c(iUAV,iUE)-const_c(iUAV,iUE))<=z(iUAV,iUE)];
        end
    end
    
      for iUE=1:K
          for iUAV=1:U
          constraints = [constraints, c(iUAV,iUE)>=0.1];
          constraints = [constraints, lambda(iUAV,iUE)<=c(iUAV,iUE)*PUAV];
% %         temp = [];
% %         temp =[indi_support_func(v,xi,delta,K,iUE),(indi_rate_lower_ue(v,delta,const_v,const_delta,H_channel,N0,K,xi,iUE,Rmin)-1)/2];
% %         
% %         temp1 =(indi_rate_lower_ue(v,delta,const_v,const_delta,H_channel,N0,K,xi,iUE,Rmin)+1)/2;
% %          constraints = [constraints, cone(temp,(indi_rate_lower_ue(v,delta,const_v,const_delta,H_channel,N0,K,xi,iUE,Rmin)+1)/2)];
          end
      end
%     
    diagnostics = optimize(constraints,obj,ops);
    testing(nIter,1) = diagnostics.problem
    opt_valu(nIter,1) = -value(obj)
    runtime= diagnostics.solvertime;
    
    const_delta = value(delta);
    const_deltab = value(deltab);
    const_c = value(c);
    const_v = value(v);
    const_p = value(p);
    
    
   % calculate achievable sum rate
   rate = zeros(K,1);
   rate_nom = zeros(K,1);
   rate_denom = zeros(K,1);
   temp = 0;
   for k=1:K
       temp = H_channel(:,k).*value(delta(:,k));
   rate_nom(k,1) = abs(value(v(:,k))'*temp)^2;
    for j=1:K
        if j~=k
            rate_denom(k,1) = rate_denom(k,1)+abs(value(v(:,j))'*temp)^2 ;
        end
    rate(k,1) = rate_nom(k,1)/(rate_denom(k,1)+N0);
    end
   end
   sum_rate(nIter,1) =sum(rate)
   end
