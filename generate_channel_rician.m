function fname=generate_channel_rician(noUAV,noUE)

%U : number of UAVs
%K: number of UEs
noAnt = 1; %number of antenna
Kb_db = 3;
Km_db = 4;
Kb = 10.^(Kb_db/10);
Km = 10.^(Km_db/10);

% coor_ue(1,1) = 4;coor_ue(1,2) = 1;
% coor_ue(2,1) = 0;coor_ue(2,2) = 1;
% coor_ue(3,1) = 2;coor_ue(3,2) = 2;
% coor_ue(4,1) = 4;coor_ue(4,2) = 4;

%Rician channel
mu = sqrt( Kb/((Kb+1)) );
s = sqrt( 1/(2*(Kb+1)) );

mu_mbs = sqrt( Km/((Km+1)) );
s_mbs = sqrt( 1/(2*(Km+1)) );

%channel from UAVs to UEs

H_channel = mu + s*(randn(noUAV,noUE) + 1j*randn(noUAV,noUE));

%channel from MBS to UAVs

for iUAV=1:noUAV
    G_channel(:,iUAV)= mu_mbs + s_mbs*(randn(noAnt,1) + 1j*randn(noAnt,1));
end

N0 = 1e-2;

filename = ['parameter_9_12.mat'];
fname=filename;
save(filename,'H_channel','G_channel','N0','noAnt');

    
end

