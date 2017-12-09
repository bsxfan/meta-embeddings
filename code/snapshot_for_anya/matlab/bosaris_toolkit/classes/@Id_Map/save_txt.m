function save_txt(idmap,outfilename)
% Saves the Id_Map to a text file.
% Inputs:
%   idmap: An object of type Id_Map.
%   outfilename: The name for the output text file.

assert(nargin==2)
assert(isa(idmap,'Id_Map'))
assert(isa(outfilename,'char'))
assert(idmap.validate(false))

outfile = fopen(outfilename,'w');

numentries = length(idmap.leftids);
for ii=1:numentries
    leftid = idmap.leftids{ii};
    rightid = idmap.rightids{ii};
    fprintf(outfile,'%s %s\n',leftid,rightid);
end

fclose(outfile);
