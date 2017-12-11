function scores = get_scores(scr,ndx)
% Inputs:
%   scr: A Scores object.
%   ndx: A Ndx object.
% Outputs:
%   scores: A vector of all valid scores.

assert(nargin==2)
assert(isa(scr,'Scores'))
assert(scr.validate())
assert(ndx.validate())

if isa(ndx,'Ndx')
    trialmask  = ndx.trialmask;
elseif isa(ndx,'Key')
    trialmask = ndx.tar | ndx.non;
end

scr = scr.align_with_ndx(ndx);
scores = scr.scoremat;
ii = trialmask & scr.scoremask;
scores = scores(ii(:))';
end
