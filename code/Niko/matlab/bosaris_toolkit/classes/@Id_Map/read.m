function idmap = read(infilename)
% Reads information from a file and constructs an Id_Map object.  The
% type of file is deduced from the extension.
% Inputs:
%   infilename: The name for the file to read.  The extension
%     (part after the final '.') must be a known type.
% Outputs:
%   idmap: An Id_Map object created from the information in the file.

assert(nargin==1)
assert(isa(infilename,'char'))

dotpos = strfind(infilename,'.');
assert(~isempty(dotpos))
extension = infilename(dotpos(end)+1:end);
assert(~isempty(extension))

if strcmp(extension,'hdf5') || strcmp(extension,'h5')
    idmap = Id_Map.read_hdf5(infilename);
elseif strcmp(extension,'mat')
    idmap = Id_Map.read_mat(infilename);
elseif strcmp(extension,'txt')
    idmap = Id_Map.read_txt(infilename);
else
    error('Unknown extension "%s"\n',extension)
end
