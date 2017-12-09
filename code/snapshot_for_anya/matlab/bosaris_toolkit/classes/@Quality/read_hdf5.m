function qual = read_hdf5(infilename)
% Creates a Quality object from the contents of an hdf5 file.  
% Inputs:
%   infilename: The name of the hdf5 file containing the quality
%     measure data.
% Outputs:
%   qual: A Quality object encoding the information in the input
%     hdf5 file.

assert(nargin==1)
assert(isa(infilename,'char'))

qual = Quality();

qual.modelQ = hdf5read(infilename,'/modelQ','V71Dimensions',true);
qual.segQ = hdf5read(infilename,'/segQ','V71Dimensions',true);

tmp = hdf5read(infilename,'/ID/row_ids','V71Dimensions',true);
numentries = length(tmp);
qual.modelset = cell(numentries,1);
for ii=1:numentries
    qual.modelset{ii} = tmp(ii).Data;
end

tmp = hdf5read(infilename,'/ID/column_ids','V71Dimensions',true);
numentries = length(tmp);
qual.segset = cell(numentries,1);
for ii=1:numentries
    qual.segset{ii} = tmp(ii).Data;
end

nummodels = length(qual.modelset);
numsegs = length(qual.segset);

colrowsel = false;
info = hdf5info(infilename);
datasets = info.GroupHierarchy.Groups.Datasets;
for ii=1:length(datasets)
    if strcmp(datasets(ii).Name,'/ID/has_row')
	colrowsel = true;
    end
end

if colrowsel
    qual.hasmodel = logical(hdf5read(infilename,'/ID/has_row'));
    qual.hasseg = logical(hdf5read(infilename,'/ID/has_column'));  
else
    qual.hasmodel = true(1,nummodels);
    qual.hasseg = true(1,numsegs);
end

qual.scoremask = true(nummodels,numsegs);

assert(qual.validate())
