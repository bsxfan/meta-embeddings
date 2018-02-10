function count = count_Gibbs_states(m,k)
% m: number of new labels
% k: number of enrolled speakers

count = 0;
ii = zeros(1,m+1);
done = false;
while ~done
   new = sum(ii==0)-1;
   count = count + Bell(new);
   ii(1) = ii(1) + 1;
   for i=1:m
       if ii(i)>k
           ii(i) = 0;
           ii(i+1) = ii(i+1) + 1;
       end
       if ii(m+1)>0
           done = true;
       end
   end
end



end