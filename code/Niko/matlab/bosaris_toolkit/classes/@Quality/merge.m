function qual = merge(qual1,qual2)
% Merges two Quality objects.  The modelsets and segsets of the two
% input objects must be disjoint.
% Inputs:
%   qual1: A Quality object.
%   qual2: Another Quality object.
% Outputs:
%   qual: A Quality object that contains the information from the two
%     input objects.

assert(nargin==2)
assert(isa(qual1,'Quality'))
assert(isa(qual2,'Quality'))
assert(qual1.validate())
assert(qual2.validate())

assert(isempty(intersect(qual1.modelset,qual2.modelset)))
assert(isempty(intersect(qual1.segset,qual2.segset)))

qual = Quality();
qual.modelset = {qual1.modelset{:}, qual2.modelset{:}};
qual.segset = {qual1.segset{:}, qual2.segset{:}};

qual.modelQ = [qual1.modelQ,qual2.modelQ];
qual.segQ = [qual1.segQ,qual2.segQ];

if ~isempty(qual1.scoremask) && ~isempty(qual2.scoremask)
    qual.scoremask = [qual1.scoremask false(length(qual1.modelset),length(qual2.segset)); false(length(qual2.modelset),length(qual1.segset)) qual2.scoremask];
else
    qual.scoremask = [];
end

if ~isempty(qual1.hasmodel) && ~isempty(qual2.hasmodel)
    qual.hasmodel = [qual1.hasmodel, qual2.hasmodel];
    qual.hasseg = [qual1.hasseg, qual2.hasseg];  
end

assert(qual.validate())
