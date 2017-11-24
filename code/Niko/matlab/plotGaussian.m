function tikz = plotGaussian(mu,C,colr,c)
    
    if nargin==0
        test_this();
        return;
    end

    if isempty(C)  %assume mu is a GME
        [mu,C] = mu.get_mu_cov();
    end
    
    [V,D] = eig(C);
    
    v1 = V(:,1);
    v2 = V(:,2);
    if all(v1>=0)
        r1 = sqrt(D(1,1));
        r2 = sqrt(D(2,2));
        rotate = acos(v1(1))*180/pi;
    elseif all(-v1>=0)
        r1 = sqrt(D(1,1));
        r2 = sqrt(D(2,2));
        rotate = acos(-v1(1))*180/pi;
    elseif all(v2>=0)
        r1 = sqrt(D(2,2));
        r2 = sqrt(D(1,1));
        rotate = acos(v2(1))*180/pi;
    else
        r1 = sqrt(D(2,2));
        r2 = sqrt(D(1,1));
        rotate = acos(-v2(1))*180/pi;
    end
    
    if ~isempty(colr)
        tikz = sprintf('\\draw[rotate around ={%4.3g:(%4.3g,%4.3g)},%s] (%4.3g,%4.3g) ellipse [x radius=%4.3g, y radius=%4.3g];\n',rotate,mu(1),mu(2),colr,mu(1),mu(2),r1,r2);
        fprintf('%s',tikz);
    end
    
    theta = (0:100)*2*pi/100;
    circle = [cos(theta);sin(theta)];
    ellipse = bsxfun(@plus,mu,V*sqrt(D)*circle);
    plot(ellipse(1,:),ellipse(2,:),c);
    
    
    


end

function test_this
   close all;

   %B = 2*eye(2) + ones(2);
   B = 2*eye(2) + [1,-1;-1,1];
   mu = [1;2];
   
   figure;hold;
   axis('equal');
   axis('square');
   plotGaussian(mu,B,'blue','b')
end