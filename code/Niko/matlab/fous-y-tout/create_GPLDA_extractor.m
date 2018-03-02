function PLDA = create_GPLDA_extractor(F,W)

    if nargin==0
        test_this();
        return;
    end

    [rdim,zdim] = size(F);
    assert(rdim>zdim);
    
    if ~exist('W','var') || isempty(W)
        W = speye(rdim);
    end
    
    E = F.'*W*F;

    SGME = create_SGME_calculator(E);
    
    V = SGME.V;  % E = VDV'
    VFW = V.'*F.'*W;
    
    PLDA.extractSGMEs = @extractSGMEs;
    PLDA.SGME = SGME;
    PLDA.getPd = @getPd;
    
    function [P,d] = getPd()
        P = VFW;
        d = SGME.d;
    end
    
    
    function [A,b] = extractSGMEs(R)
        b = ones(1,size(R,2));
        A = VFW*R;
    end
    

end

function test_this()

  error('test_this not implemented');    
    
end

