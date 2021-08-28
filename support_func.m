function result = support_func(v,xi,delta,noUE)

temp0 = 0;
 for index=1:noUE
     temp = 0;
     for i=1:noUE
         temp = temp + v(:,i)'*v(:,i);
     end
     temp = temp + delta(:,index)'*delta(:,index);
     
     temp0 =  temp0 + xi(index,1)*temp;
 end
 
 result = temp0;
end