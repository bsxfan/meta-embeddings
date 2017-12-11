function PLDA = create_TPLDA(F,W,nu)

    [D,d] = size(F);
    assert(D>d,'D>d required');
    assert(nu>0,'nu>0 required');
    
    FW = F.'*W;
    B = FW*F;
    
    [V,L] = eig(B);  % B = V*L*V'  % inv(B) = V*inv(L)*V'
    lambda = diag(L);
    assert(all(lambda>0),'F''WF should be invertible');
    VFW = V.'*FW;
    
    G = W - VFW.'*bsxfun(@ldivide,lambda,VFW);
    
    
    PLDA.extract = @extract;
    
    
    function GME = extract(X)
        beta = (nu+D-d)./(nu+sum(X.'*G*X,1));
        GME = [FW*X;beta];
    end
    
    
    function 


end