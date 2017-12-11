classdef Quality
% A class for storing quality measure information.  Quality
% measures are segment (not trial) based.  The class stores vectors
% of quality measures for each model (train segment) in modelQ and
% for each test segment in segQ.

properties
  modelQ
  segQ
  modelset
  segset
  hasmodel
  hasseg
  scoremask
end

methods
  % qual = Quality()
  % qual = Quality(quality_struct)
  % qual = Quality(model_list,seg_list,model_quality_values,seg_quality_values)
  % qual = Quality(model_list,seg_list,model_quality_values,seg_quality_values,has_model,has_seg)
  % qual = Quality(model_list,seg_list,model_quality_values,seg_quality_values,has_model,has_seg,trial_mask)
  function qual = Quality(param1,param2,param3,param4,param5,param6,param7)
    switch nargin
     case 0
      qual.modelQ = [];
      qual.segQ = [];
      qual.modelset = {};
      qual.segset = {}; 
      qual.hasmodel = logical([]);
      qual.hasseg = logical([]);
      qual.scoremask = logical([]);
     case 1
      assert(isstruct(param1))
      assert(iscell(param1.modelset))
      assert(iscell(param1.segset))
      qual.modelset = param1.modelset;
      qual.segset = param1.segset;
      qual.modelQ = param1.modelQ;
      qual.segQ = param1.segQ;
      if isfield(param1,'hasmodel')
	qual.hasmodel = param1.hasmodel;
	qual.hasseg = param1.hasseg;
      else
	qual.hasmodel = true(1,length(param1.modelset));
	qual.hasseg = true(1,length(param1.segset));
      end
      if isfield(param1,'scoremask')
	qual.scoremask = param1.scoremask;
      else
	qual.scoremask = true(length(param1.modelset),length(param1.segset));
      end
     case 4
      assert(iscell(param1))
      assert(iscell(param2))
      qual.modelset = param1;
      qual.segset = param2;
      qual.modelQ = param3;
      qual.segQ = param4;
      qual.hasmodel = true(1,length(param1));
      qual.hasseg = true(1,length(param2));
      qual.scoremask = true(length(param1),length(param2));
     case 6
      assert(iscell(param1))
      assert(iscell(param2))
      qual.modelset = param1;
      qual.segset = param2;
      qual.modelQ = param3;
      qual.segQ = param4;
      qual.hasmodel = param5;
      qual.hasseg = param6;
      qual.scoremask = true(length(param1),length(param2));
     case 7
      assert(iscell(param1))
      assert(iscell(param2))
      qual.modelset = param1;
      qual.segset = param2;
      qual.modelQ = param3;
      qual.segQ = param4;
      qual.hasmodel = param5;
      qual.hasseg = param6;
      qual.scoremask = param7;
     otherwise
      error('Incorrect number of parameters in constructor.  Should be 0, 1, 4, 6 or 7.')
    end
  end
end

methods
  save(qual,outfilename)
  save_hdf5(qual,outfilename)
  save_mat(qual,outfilename)
  qual = align_with_ndx(qual,ndx)
  ok = validate(qual)
end

methods (Static)
  qual = read(infilename)
  qual = read_hdf5(infilename)
  qual = read_mat(infilename)
  qual = merge(qual1,qual2) 
end

end
