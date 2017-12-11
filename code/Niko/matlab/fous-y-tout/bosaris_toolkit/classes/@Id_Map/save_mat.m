function save_mat(idmap,outfilename)
% Saves the Id_Map object in a mat file.
% Inputs:
%   idmap: The Id_Map object to save.
%   outfilename: The name of the output mat file.

assert(nargin==2)
assert(isa(idmap,'Id_Map'))
assert(isa(outfilename,'char'))
assert(idmap.validate(false))

warning('off','MATLAB:structOnObject')
idmap = struct(idmap);
warning('on','MATLAB:structOnObject')
save(outfilename,'idmap');
