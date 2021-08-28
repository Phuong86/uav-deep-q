function result = support_func_c(v,zeta,delta,noUE,c,index_uav)

temp0 = 0;
 for index=1:noUE
     temp = 0;
     for i=1:noUE
         temp = temp + v(:,i)'*v(:,i);
     end
     temp = temp + delta(:,index)'*delta(:,index)+c(index_uav,index)^2;
     
     temp0 =  temp0 + zeta(index,1)*temp;
 end
 
 result = temp0;
end