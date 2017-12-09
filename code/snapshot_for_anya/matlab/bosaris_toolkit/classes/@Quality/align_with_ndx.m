function aligned_qual = align_with_ndx(qual,ndx)
% The ordering in the output Quality object corresponds to ndx, so
% aligning several Quality objects with the same ndx will result in
% them being comparable with each other.
% Inputs:
%   qual: a Quality object
%   ndx: a Key or Ndx object
% Outputs:
%   aligned_qual: qual resized to size of 'ndx' and reordered
%     according to the ordering of modelset and segset in 'ndx'.

assert(nargin==2)
assert(isa(qual,'Quality'))
assert(isa(ndx,'Key')||isa(ndx,'Ndx'))
assert(qual.validate())
assert(ndx.validate())

aligned_qual = Quality();

aligned_qual.modelset = ndx.modelset;
aligned_qual.segset = ndx.segset;
m = length(ndx.modelset);
n = length(ndx.segset);


[hasmodel,rindx] = ismember(ndx.modelset,qual.modelset);
rindx = rindx(hasmodel);
[hasseg,cindx] = ismember(ndx.segset,qual.segset);
cindx = cindx(hasseg);

aligned_qual.hasmodel = hasmodel;
aligned_qual.hasseg = hasseg;

if any(ismember(properties(qual),'hasmodel'))
    aligned_qual.hasmodel(hasmodel) = qual.hasmodel(rindx);
    aligned_qual.hasseg(hasseg) = qual.hasseg(cindx);
end

q = size(qual.modelQ,1);
  
assert(q==size(qual.segQ,1));
assert(all(isfinite(qual.modelQ(:))));
assert(all(isfinite(qual.segQ(:))));
  
aligned_qual.modelQ = zeros(q,m);
aligned_qual.modelQ(:,hasmodel) = double(qual.modelQ(:,rindx)); 
    
aligned_qual.segQ = zeros(q,n);
aligned_qual.segQ(:,hasseg) = double(qual.segQ(:,cindx)); 

aligned_qual.scoremask = false(m,n);
aligned_qual.scoremask(hasmodel,hasseg) = qual.scoremask(rindx,cindx);

assert(sum(aligned_qual.scoremask(:)) <= sum(aligned_qual.hasmodel)*sum(aligned_qual.hasseg));


if isa(ndx,'Ndx')
    aligned_qual.scoremask = aligned_qual.scoremask & ndx.trialmask;
else
    aligned_qual.scoremask = aligned_qual.scoremask & (ndx.tar | ndx.non);
end

if sum(aligned_qual.hasmodel)<m
    log_warning('models reduced from %i to %i\n',m,sum(aligned_qual.hasmodel));
end
if sum(aligned_qual.hasseg)<n
    log_warning('testsegs reduced from %i to %i\n',n,sum(aligned_qual.hasseg));
end

if isa(ndx,'Key') %supervised
    tar = ndx.tar & aligned_qual.scoremask;
    non = ndx.non & aligned_qual.scoremask;
    
    missing = sum(ndx.tar(:)) - sum(tar(:));
    if missing > 0
	log_warning('%i of %i targets missing.\n',missing,sum(ndx.tar(:)));
    end
    missing = sum(ndx.non(:)) - sum(non(:));
    if missing > 0
	log_warning('%i of %i non-targets missing.\n',missing,sum(ndx.non(:)));
    end
else
    mask = ndx.trialmask & aligned_qual.scoremask;
    
    missing = sum(ndx.trialmask(:)) - sum(mask(:));
    if missing > 0
	log_warning('%i of %i trials missing\n',missing,sum(ndx.trialmask(:)));
    end

end

assert(aligned_qual.validate())

end
