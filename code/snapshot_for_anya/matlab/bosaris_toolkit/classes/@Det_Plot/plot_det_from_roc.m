function plot_det_from_roc(plot_obj,Pmiss,Pfa,plot_args,legend_string)
% Plots a DET curve.  
% Inputs:
%   plot_args: A cell array of arguments to be passed to plot that control
%     the appearance of the curve. See Matlab's help on 'plot' for information.
%   legend_string: Optional.  A string to describe this curve in the legend.

if ischar(plot_args)
    plot_args = {plot_args};
end

figure(plot_obj.fh);


x = probit(Pfa);
y = probit(Pmiss);
assert(iscell(plot_args))
lh = plot(x,y,plot_args{:});

if exist('legend_string','var') && ~isempty(legend_string)
    assert(ischar(legend_string))
    plot_obj.add_legend_entry(lh,legend_string,true);
end

end
