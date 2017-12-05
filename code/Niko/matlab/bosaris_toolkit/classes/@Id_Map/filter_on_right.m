function out_idmap = filter_on_right(in_idmap,idlist,keep)
% Removes some of the information in an idmap.  Depending on the
% value of 'keep', the idlist indicates the strings to retain or
% the strings to discard.
% Inputs:
%   in_idmap: An Id_Map object to be pruned.
%   idlist: A cell array of strings which will be compared with
%     the rightids of 'in_idmap'.
%   keep: A boolean indicating whether idlist contains the ids to
%     keep or to discard. 
% Outputs:
%   out_idmap: A filtered version of 'in_idmap'.

if nargin == 0
    test_this();
    return
end

assert(nargin==3)
assert(isa(in_idmap,'Id_Map'))
assert(in_idmap.validate())
assert(iscell(idlist))

if keep
    keepids = idlist;
else
    keepids = setdiff(in_idmap.rightids,idlist);
end

keep_idx = ismember(in_idmap.rightids,keepids);

out_idmap = Id_Map();
out_idmap.leftids = in_idmap.leftids(keep_idx);
out_idmap.rightids = in_idmap.rightids(keep_idx);
assert(out_idmap.validate(false))

end

function test_this()

idmap = Id_Map();
idmap.leftids = {'aaa','bbb','ccc','bbb','ddd','eee'};
idmap.rightids = {'11','22','33','44','55','22'};

fprintf('idmap.leftids\n');
disp(idmap.leftids)
fprintf('idmap.rightids\n');
disp(idmap.rightids)

idlist = {'22','44'}

keep = true

out = Id_Map.filter_on_right(idmap,idlist,keep);

fprintf('out.leftids\n');
disp(out.leftids)
fprintf('out.rightids\n');
disp(out.rightids)

keep = false

out = Id_Map.filter_on_right(idmap,idlist,keep);

fprintf('out.leftids\n');
disp(out.leftids)
fprintf('out.rightids\n');
disp(out.rightids)

idlist = {'11','33','66'}

keep = true

out = Id_Map.filter_on_right(idmap,idlist,keep);

fprintf('out.leftids\n');
disp(out.leftids)
fprintf('out.rightids\n');
disp(out.rightids)

keep = false

out = Id_Map.filter_on_right(idmap,idlist,keep);

fprintf('out.leftids\n');
disp(out.leftids)
fprintf('out.rightids\n');
disp(out.rightids)

end
