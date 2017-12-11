function idmap = read_mat(infilename)
% Reads a struct from a mat file and constructs an Id_Map object.
% Inputs:
%   infilename: The name for the mat file to read.
% Outputs:
%   idmap: An Id_Map object created from the information in the mat
%     file.

assert(nargin==1)
assert(isa(infilename,'char'))

load(infilename,'idmap');
idmap = Id_Map(idmap);

assert(idmap.validate())
