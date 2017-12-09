function idmap = read_hdf5(infilename)
% Creates an Id_Map object from the information in an hdf5 file.
% Inputs:
%   infilename: The name of the hdf5 file containing the information
%     necessary to construct an Id_Map object.
% Outputs:
%   idmap: An Id_Map object containing the information in the input
%     file.

assert(nargin==1)
assert(isa(infilename,'char'))

idmap = Id_Map();
idmap.leftids = h5strings_to_cell(infilename,'/left_ids');
idmap.rightids = h5strings_to_cell(infilename,'/right_ids');
assert(idmap.validate())

function cellstrarr = h5strings_to_cell(infilename,attribname)
tmp = hdf5read(infilename,attribname,'V71Dimensions',true);
numentries = length(tmp);
cellstrarr = cell(numentries,1);
for ii=1:numentries
    cellstrarr{ii} = tmp(ii).Data;
end
