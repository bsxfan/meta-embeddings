function save_mat(qual,outfilename)
% Saves a Quality object to a mat file.
% Inputs:
%   qual: The Quality object to be saved.
%   outfilename: The name for the mat output file.

assert(nargin==2)
assert(isa(qual,'Quality'))
assert(isa(outfilename,'char'))
assert(qual.validate())

warning('off','MATLAB:structOnObject')
qual = struct(qual);
warning('on','MATLAB:structOnObject')
save(outfilename,'qual');
