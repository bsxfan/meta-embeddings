function save_hdf5(idmap,outfilename)
% Saves the Id_Map object to an hdf5 file.
% Inputs:
%   idmap: The Id_Map object to save.
%   outfilename: The name of the output hdf5 file.

assert(nargin==2)
assert(isa(idmap,'Id_Map'))
assert(isa(outfilename,'char'))
assert(idmap.validate(false))

hdf5write(outfilename,'/left_ids',idmap.leftids,'V71Dimensions',true);
hdf5write(outfilename,'/right_ids',idmap.rightids,'V71Dimensions',true,'WriteMode','append');
