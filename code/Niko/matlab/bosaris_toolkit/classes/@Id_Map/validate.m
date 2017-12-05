function ok = validate(idmap,warn)
% Checks that an object of type Id_Map obeys certain rules that
% must alows be true.
% Inputs:
%   idmap: the object to be checked.
%   warn: boolean.  Indicates whether to print a warning about
%     duplicate strings.
% Outputs:
%   ok: a boolean value indicating whether the object is valid.

assert((nargin==1)||(nargin==2))
assert(isa(idmap,'Id_Map'))

if ~exist('warn','var')
    warn = true;
end

ok = iscell(idmap.leftids);
ok = ok && iscell(idmap.rightids);
ok = ok && length(idmap.leftids)==length(idmap.rightids);
ok = ok && size(idmap.leftids,2)==1;
ok = ok && size(idmap.rightids,2)==1;

if warn
    if length(idmap.leftids) ~= length(unique(idmap.leftids))
	log_warning('The left id list contains duplicate identifiers.\n')      
    end
    if length(idmap.rightids) ~= length(unique(idmap.rightids))
	log_warning('The right id list contains duplicate identifiers.\n')      
    end
end
