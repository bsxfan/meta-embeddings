function model = SGME_train(R,labels,zdim,nu,niters,timeout)

    rdim = size(R,1);
    delta = rdim - zdim;

    if isempty(nu)  %learn nu
        model.getParams = @init2;
        model.extract = @(w,R) SGME_extract(w,zdim,R);
    else %plug in nu
        model.getParams = @init1;
        model.extract = @(w,R) SGME_extract(w,zdim,R,nu);
    end
    model.logexpectations = @SGME_logexpectations;


    model = addME_train(R,labels,model,niters,timeout);

    
    
    function [w_ext,w_exp] = init1()
        P = randn(zdim,rdim);
        H = randn(delta,rdim);
        w_ext = [P(:);H(:)];
        w_exp = randn(zdim,1);
    end
    
    function [w_ext,w_exp] = init2()
        nu = pi;
        w1 = log(nu);
        P = randn(zdim,rdim);
        H = randn(delta,rdim);
        w_ext = [w1;P(:);H(:)];
        w_exp = randn(zdim,1);
    end
    
    
    
end