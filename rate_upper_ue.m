function result = rate_upper_ue(v,delta,const_v,const_delta,channel_UAV_UE,N0,noUE,zeta,c,const_c,index_uav)

result_temp = 0;

for index=1:noUE
    temp_rate_denom = 0;
    
    temp_denom0 = 0;
    temp_num0 = 0;
    
    temp_denom1 = 0;
    temp_num1 = 0;
    
    temp2 = 0;
    temp_beam = 0;
    
    channel_PL_UAV_UE = channel_UAV_UE(:,index).*const_delta(:,index);
    mat_channel = channel_PL_UAV_UE*channel_PL_UAV_UE';

    bre_channel_PL_UAV_UE = 0;
    bre2_channel_PL_UAV_UE = 0;
    
    for i=1:noUE
        bre_channel_PL_UAV_UE = bre_channel_PL_UAV_UE + (const_v(:,i).*channel_UAV_UE(:,index))*(const_v(:,i).*channel_UAV_UE(:,index))';
        if i ~= index
            bre2_channel_PL_UAV_UE = bre2_channel_PL_UAV_UE + (const_v(:,i).*channel_UAV_UE(:,index))*(const_v(:,i).*channel_UAV_UE(:,index))';
        end
    end
    
    %first term of f hat
    for i=1:noUE
    temp_denom0 = temp_denom0 + abs(const_v(:,i)'*channel_PL_UAV_UE)^2;
    end
    temp_denom0 = temp_denom0 + N0;
    
    for i=1:noUE
    temp_num0 = temp_num0 + 2*real(const_v(:,i)'*mat_channel*v(:,i) - const_v(:,i)'*mat_channel*const_v(:,i)) ;
    end
    first_frac = temp_num0/temp_denom0;
    
    %second term of f hat
    for i=1:noUE
    if i ~= index
        temp_denom1 = temp_denom1 + abs(const_v(:,i)'*channel_PL_UAV_UE)^2;
    end
    end
    temp_denom1 = temp_denom1 + N0;
    
    for i=1:noUE
    if i ~= index
        temp_num1 = temp_num1 + 2*real(const_v(:,i)'*mat_channel*v(:,i) - const_v(:,i)'*mat_channel*const_v(:,i));
    end
    end
    second_frac = temp_num1/temp_denom1;
    
    
    %first term of f breve
    
    temp_denom0_breve = real(const_delta(:,index)'*bre_channel_PL_UAV_UE*const_delta(:,index)) + N0;
    temp_num0_breve = 2*real(const_delta(:,index)'*bre_channel_PL_UAV_UE*delta(:,index) - const_delta(:,index)'*bre_channel_PL_UAV_UE*const_delta(:,index));
    third_frac = temp_num0_breve/temp_denom0_breve;
    
    %second term of f breve
    
    temp_denom1_breve = real(const_delta(:,index)'*bre2_channel_PL_UAV_UE*const_delta(:,index)) + N0;
    temp_num1_breve = 2*real(const_delta(:,index)'*bre2_channel_PL_UAV_UE*delta(:,index) - const_delta(:,index)'*bre2_channel_PL_UAV_UE*const_delta(:,index));
    forth_frac = temp_num1_breve/temp_denom1_breve;
    
    temp_rate_num = abs(const_v(:,index)'*channel_PL_UAV_UE)^2;
    
    for i=1:noUE
        if i~=index
            temp_rate_denom = temp_rate_denom + abs(const_v(:,i)'*channel_PL_UAV_UE)^2;
        end
    end
    
    temp_rate = const_c(index_uav,index)*log(1+temp_rate_num/(temp_rate_denom+N0));
    
    %term of approximation of variable c
    
    temp_c = (c(index_uav,index)-const_c(index_uav,index))*temp_rate;
    
    % Term quad approximate
    
    for i=1:noUE
    temp2 = temp2 + 2*real(const_v(:,i)'*v(:,i) - const_v(:,i)'*const_v(:,i));
    end
    temp2 = temp2 + 2*(const_delta(:,index)'*delta(:,index) - const_delta(:,index)'*const_delta(:,index))+2*(const_c(index_uav,index)*c(index_uav,index)-const_c(index_uav,index)^2);
% % Term quad constant

    for i=1:noUE
    temp_beam = temp_beam + const_v(:,i)'*const_v(:,i);
    end
    temp_beam = temp_beam + const_delta(:,index)'*const_delta(:,index)+const_c(index_uav,index)^2;
    
    result_temp = result_temp + temp_rate -  zeta(index,1)*temp_beam + const_c(index_uav,index)*first_frac - const_c(index_uav,index)*second_frac + const_c(index_uav,index)*third_frac - const_c(index_uav,index)*forth_frac - zeta(index,1)*temp2 +temp_c;
end
result = result_temp;
end