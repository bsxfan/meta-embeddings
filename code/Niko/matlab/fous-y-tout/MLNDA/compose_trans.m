function [f,fi,paramsz] = compose_trans(outer_paramsz,outer_f,outer_fi,inner_paramsz,inner_f,inner_fi)

    if nargout==0
        test_this();
        return;
    end



    f = @f_this;
    fi = @fi_this;
    paramsz = outer_paramsz + inner_paramsz;
    
    function T = f_this(P,R)
        [outerP,innerP] = unpack(P);
        T = outer_f(outerP,inner_f(innerP,R));
    end
    
    function [X,logdetJ,back] = fi_this(P,Y)
        [outerP,innerP] = unpack(P);
        [Z,logdet1,back1] = outer_fi(outerP,Y);
        [X,logdet2,back2] = inner_fi(innerP,Z);
        logdetJ = logdet1 + logdet2;
        
        back = @back_this;
        
        
        function [dParams,dT] = back_this(dR,dlogdetJ)
            [dParam2,dT] = back2(dR,dlogdetJ);
            if nargout>=2
                [dParam1,dT] = back1(dT,dlogdetJ);
            else
                [dParam1] = back1(dT,dlogdetJ);
            end
            dParams = [dParam1(:);dParam2(:)];
        end
        
        
    end


    function [outerP,innerP] = unpack(P)
        outerP = P(1:outer_paramsz);
        innerP = P(outer_paramsz+1:paramsz);
    end


end

function test_this()

    dim = 3;

    [inner_f,inner_fi,szi] = create_affineTrans(dim);
    [outer_f,outer_fi,szo] = create_affineTrans(dim);
    
    [f,fi,sz] = compose_trans(szo,outer_f,outer_fi,szi,inner_f,inner_fi);
    
    n = 5;
    P = randn(sz,1);
    
    R = randn(dim,n);
    T = f(P,R);
    
    testBackprop_multi(fi,2,{P,T});
    
    
    


end

