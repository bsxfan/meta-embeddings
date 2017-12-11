function idmap = merge(idmap1,idmap2)
% Merges two Id_Map objects.  idmap2 is appended to idmap1.
% Inputs:
%   idmap1: An Id_Map object.
%   idmap2: Another Id_Map object.
% Outputs:
%   idmap: An Id_Map object that contains the information from the two
%     input Id_Maps. 

assert(nargin==2)
assert(isa(idmap1,'Id_Map'))
assert(isa(idmap2,'Id_Map'))
assert(idmap1.validate())
assert(idmap2.validate())

if ~isempty(intersect(idmap1.leftids,idmap2.leftids)) || ~isempty(intersect(idmap1.rightids,idmap2.rightids))
    log_warning('idmaps being merged share ids.\n')
end
    
idmap = Id_Map();
idmap.leftids = {idmap1.leftids{:}, idmap2.leftids{:}}';
idmap.rightids = {idmap1.rightids{:}, idmap2.rightids{:}}';

assert(idmap.validate(false))
