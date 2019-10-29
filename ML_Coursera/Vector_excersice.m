%Create a row vector 

k=8;
%A for loop strategy
vk= zeros(1,k+1);
for j= 0:k
   vk(j+1) = (2*j+1)^2;
end

vk

