function result = rate_lower_uav(p,const_p,deltab,const_deltab,psi,index_uav,channel_mbs_uav,N0)


channel_PL_mbs_uav = channel_mbs_uav(1,index_uav)*const_deltab(index_uav,1);
mat_channel_uav = channel_mbs_uav(1,index_uav)'*channel_mbs_uav(1,index_uav);
temp_rate = log(1+(const_p(index_uav,1)*channel_PL_mbs_uav'*channel_PL_mbs_uav)/N0);

% % Term quad constant

temp_beam = psi(index_uav,1)*(const_p(index_uav,1)^2+const_deltab(index_uav,1)^2);

%% % Term quad approximate

temp2 = 2*psi(index_uav,1)*(p(index_uav,1)*const_p(index_uav,1)-const_p(index_uav,1)^2 + const_deltab(index_uav,1)*deltab(index_uav,1)-const_deltab(index_uav,1)^2);

%%% approximate to power uav p

first_frac_num = channel_PL_mbs_uav'*channel_PL_mbs_uav*(p(index_uav,1)-const_p(index_uav,1));
first_frac_denum = N0+const_p(index_uav,1)*channel_PL_mbs_uav'*channel_PL_mbs_uav;

first_frac = first_frac_num/first_frac_denum;

%%% approximate to location uav deltab

second_frac_num = 2*const_p(index_uav,1)*real(const_deltab(index_uav,1)*mat_channel_uav*deltab(index_uav,1)-channel_PL_mbs_uav'*channel_PL_mbs_uav);
second_frac = second_frac_num/first_frac_denum;

result = temp_rate + temp_beam + temp2 + first_frac + second_frac;




end