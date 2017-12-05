function ok = validate(qual)
% Checks that an object of type Quality obeys certain rules that
% must always be true.
% Inputs:
%   qual: the object to be checked.
% Outputs:
%   ok: a boolean value indicating whether the object is valid.

assert(nargin==1)
assert(isa(qual,'Quality'))

ok = iscell(qual.modelset);
ok = ok && iscell(qual.segset);

nummods = length(qual.modelset);
numsegs = length(qual.segset);

ok = ok && (size(qual.scoremask,1)==nummods);
ok = ok && (size(qual.scoremask,2)==numsegs);

ok = ok && (size(qual.modelQ,1)==size(qual.segQ,1));

ok = ok && (size(qual.modelQ,2)==nummods);
ok = ok && (size(qual.segQ,2)==numsegs);

ok = ok && (length(qual.hasmodel)==nummods);
ok = ok && (length(qual.hasseg)==numsegs);
