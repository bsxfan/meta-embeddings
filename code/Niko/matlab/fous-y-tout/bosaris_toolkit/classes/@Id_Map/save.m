function save(idmap,outfilename)
% Saves an Id_Map object to a file.  The file type is determined by
% the extension.
% Inputs:
%   idmap: The Id_Map object to be saved.
%   outfilename: The name for the output file.

assert(nargin==2)
assert(isa(idmap,'Id_Map'))
assert(isa(outfilename,'char'))
assert(idmap.validate(false))

dotpos = strfind(outfilename,'.');
assert(~isempty(dotpos))
extension = outfilename(dotpos(end)+1:end);
assert(~isempty(extension))

if strcmp(extension,'hdf5') || strcmp(extension,'h5')
    idmap.save_hdf5(outfilename);
elseif strcmp(extension,'mat')
    idmap.save_mat(outfilename);
elseif strcmp(extension,'txt')
    idmap.save_txt(outfilename);  
else
    error('Unknown extension "%s"\n',extension)
end
