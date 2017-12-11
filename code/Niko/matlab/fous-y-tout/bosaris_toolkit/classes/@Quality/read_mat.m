function qual = read_mat(infilename)
% Creates a Quality object from the information in a mat file.
% Inputs:
%   infilename: A mat file containing quality measures.
% Outputs:
%   qual: A Quality object encoding the quality measures in the
%     input mat file. 

assert(nargin==1)
assert(isa(infilename,'char'))

load(infilename,'qual');
qual = Quality(qual);

assert(qual.validate())
