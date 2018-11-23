function [model,X] = assemble_truncDPMM(alpha,F,W,m,n)
%Model is w -> hlabels -> X <- Z   



    %diagonalize model
    E = F.'*W*F;
    [V,~] = eig(E);
    F = F*V;


    [D,d] = size(F);

    w_node = create_symmetricDirichlet_node(log(alpha),m);
    w0 = w_node.get();
    
    L_node = create_Label_node(log(w0),n);
    hlabels0 = L_node.get();
    
    Z_node = create_stdMVG_node(d,m);
    Z0 = Z_node.get();
    
    X_node = create_SPLDA_node(hlabels0,Z0,F,W);
    X = X_node.get();
    
    
    model.fullGibbs_iteration = @fullGibbs_iteration;
    
    
    
    function hlabels = fullGibbs_iteration(hlabels)
        if nargin>0
            L_node.observe(hlabels);
        else
            hlabels = L_node.get();
        end
        
        %go right
        X_node.condition_on_parent(1,hlabels);
        Z_node.condition_on_child(X_node.inferParent(2));
        
        %sample Z and bounce back left
        X_node.condition_on_parent(2,Z_node.sample());
        L_node.condition_on_child(X_node.inferParent(1));

        %go left
        w_node.condition_on_child(L_node.inferParent());
        
        %sample w and bounce back right
        L_node.condition_on_parent(w_node.sample());

        
        %update
        hlabels = L_node.sample();
        
    end



end