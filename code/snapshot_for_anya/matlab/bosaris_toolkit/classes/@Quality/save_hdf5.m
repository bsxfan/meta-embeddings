function save_hdf5(qual,outfilename)
% Saves a Quality object to an hdf5 file.
% Inputs:
%   qual: The Quality object to be saved.
%   outfilename: The name for the hdf5 output file.

assert(nargin==2)
assert(isa(qual,'Quality'))
assert(isa(outfilename,'char'))
assert(qual.validate())

scoremask = uint8(qual.scoremask);
modelset = cellstr(qual.modelset);
segset = cellstr(qual.segset);

hdf5write(outfilename,'/modelQ',qual.modelQ,'V71Dimensions',true);
hdf5write(outfilename,'/segQ',qual.segQ,'V71Dimensions',true,'WriteMode','append');
hdf5write(outfilename,'/score_mask',scoremask,'V71Dimensions',true,'WriteMode','append');
hdf5write(outfilename,'/ID/row_ids',modelset,'V71Dimensions',true,'WriteMode','append');
hdf5write(outfilename,'/ID/column_ids',segset,'V71Dimensions',true,'WriteMode','append');

if any(ismember(properties(qual),'hasmodel'))
    hasmodel = uint8(qual.hasmodel);
    hasseg = uint8(qual.hasseg);
    hdf5write(outfilename,'/ID/has_row',hasmodel,'V71Dimensions',true,'WriteMode','append');
    hdf5write(outfilename,'/ID/has_column',hasseg,'V71Dimensions',true,'WriteMode','append');  
end
