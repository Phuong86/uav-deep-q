function fname_ue=generate_ue(noUE)

for i=1:noUE
    coor_ue(i,1)=200*rand(1);%20*rand(1)
    coor_ue(i,2) = 200*rand(1);
end

filename = ['parameter_ue_location_5_200m.mat'];
fname_ue=filename;
save(filename,'coor_ue');
end