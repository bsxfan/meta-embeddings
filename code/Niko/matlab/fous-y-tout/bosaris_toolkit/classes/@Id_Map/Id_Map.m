classdef Id_Map
% A class that stores a map between identifiers (strings).  One
% list is called 'leftids' and the other 'rightids'.  The class
% provides methods that convert a sequence of left ids to a
% sequence of right ids and vice versa.  If 'leftids' or 'rightids'
% contains duplicates then all occurrences are used as the index
% when mapping.

properties
  leftids
  rightids
end

methods
  % idmap = Id_Map()
  % idmap = Id_Map(idmap_struct)
  % idmap = Id_Map(left_ids,right_ids)
  function idmap = Id_Map(param1,param2)
  switch nargin
   case 0
    idmap.leftids = {};
    idmap.rightids = {};
   case 1
    assert(isstruct(param1))
    assert(iscell(param1.leftids))
    assert(iscell(param1.rightids))
    assert(length(param1.leftids)==length(param1.rightids))
    if length(param1.leftids) ~= length(unique(param1.leftids))
	log_warning('The left id list contains duplicate identifiers.\n')      
    end
    if length(param1.rightids) ~= length(unique(param1.rightids))
	log_warning('The right id list contains duplicate identifiers.\n')      
    end
    idmap.leftids = param1.leftids;
    idmap.rightids = param1.rightids;
   case 2
    assert(iscell(param1))
    assert(iscell(param2))
    assert(length(param1)==length(param2))
    if length(param1) ~= length(unique(param1))
	log_warning('The left id list contains duplicate identifiers.\n')      
    end
    if length(param2) ~= length(unique(param2))
	log_warning('The right id list contains duplicate identifiers.\n')      
    end
    idmap.leftids = param1;
    idmap.rightids = param2;
   otherwise
    error('Incorrect number of parameters in constructor.  Should be 0, 1, or 2.')   
  end
  end
end

methods
  save(idmap,outfilename)
  save_hdf5(idmap,outfilename)
  save_mat(idmap,outfilename)
  save_txt(idmap,outfilename)
  rightidlist = map_left_to_right(idmap,leftidlist)
  leftidlist = map_right_to_left(idmap,rightidlist)
  outidmap = filter_on_left(inidmap,idlist,keep)
  outidmap = filter_on_right(inidmap,idlist,keep)
  ok = validate(idmap,warn)
end

methods (Static = true)
  idmap = read(infilename)
  idmap = read_hdf5(infilename)
  idmap = read_mat(infilename)
  idmap = read_txt(infilename)
  idmap = merge(idmap1,idmap2)
end

end
