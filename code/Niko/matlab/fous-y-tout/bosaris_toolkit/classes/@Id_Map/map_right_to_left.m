function leftidlist = map_right_to_left(idmap,rightidlist)
% Maps a list of ids to a new list of ids using the given map.  The
% input ids are matched against the rightids of the map and the
% output ids are taken from the corresponding leftids of the map.
% Inputs:
%   idmap: The Id_Map giving the mapping between the input and
%     output string lists.
%   rightidlist: A list of strings to be matched against the
%     rightids of the idmap.  The leftids corresponding to these
%     rightids will be returned.
% Outputs:
%   leftidlist: A list of strings that are the mappings of the
%     strings in rightidlist. 

assert(nargin==2)
assert(isa(idmap,'Id_Map'))
assert(iscell(rightidlist))
assert(idmap.validate())

tmpmap.keySet = idmap.rightids;
tmpmap.values = idmap.leftids;

[leftidlist,is_present] = maplookup(tmpmap,rightidlist);
num_dropped = length(is_present) - sum(is_present);
if num_dropped ~= 0
    log_warning('%d ids could not be mapped because they were not present in the map.\n',num_dropped);
end

end
