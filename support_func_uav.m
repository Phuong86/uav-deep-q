function result = support_func_uav(psi,p,deltab,index_uav)
 
         
 result = psi(index_uav,1)*(p(index_uav,1)^2+deltab(index_uav,1)^2);
end