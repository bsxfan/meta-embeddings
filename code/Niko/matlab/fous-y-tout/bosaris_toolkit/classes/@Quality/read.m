function qual = read(infilename)
% Reads information from a file and constructs a Quality object.  The
% type of file is deduced from the extension.
% Inputs:
%   infilename: The name of the file to read.  The extension
%     (part after the final '.') must be a known type.
% Outputs:
%   qual: A Quality object created from the information in the file.

assert(nargin==1)
assert(isa(infilename,'char'))

dotpos = strfind(infilename,'.');
assert(~isempty(dotpos))
extension = infilename(dotpos(end)+1:end);
assert(~isempty(extension))

if strcmp(extension,'hdf5') || strcmp(extension,'h5')
    qual = Quality.read_hdf5(infilename);
elseif strcmp(extension,'mat')
    qual = Quality.read_mat(infilename);
else
    error('Unknown extension "%s"\n',extension)
end
