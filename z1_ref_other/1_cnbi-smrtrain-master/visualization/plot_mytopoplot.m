function [ h ] = plot_mytopoplot( data, chanlocs64, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

limits = [-5 5];
style = 'map';
ncontours = 6;
electrodes = 'labels';
conv = 'off';
intrad = 1.0;
intsquare = 'on';
print_colorbar = 0;
colorbar_string = '';

% process input arguments
for i=1:length(varargin)
    arg = varargin{i};
    
    if strcmp(arg,'limits')
        limits = varargin{i+1};
    elseif strcmp(arg,'style')
        style = varargin{i+1};
    elseif strcmp(arg,'ncontours')
        ncontours = varargin{i+1};
    elseif strcmp(arg,'electrodes')
        electrodes = varargin{i+1};
    elseif strcmp(arg,'conv')
        conv = varargin{i+1};
    elseif strcmp(arg,'intrad')
        intrad = varargin{i+1};        
    elseif strcmp(arg,'intsquare')
        intsquare = varargin{i+1};
    elseif strcmp(arg,'colorbar')
        print_colorbar = varargin{i+1};
    elseif strcmp(arg,'colorbar_string')
        colorbar_string = varargin{i+1};        
    end
    
end
hold on;

% convChans(data)
h = topoplot( ...
    plot_myConvChans( data ), chanlocs64, ...
    'maplimits', limits, ...
    'style', style, ...
    'numcontour', ncontours, ...
    'electrodes', 'off', ...
    'headrad', 'rim', ...
    'conv', conv, ...
    'intrad', intrad, ...
    'intsquare', intsquare ...     
    );

topoplot( ...
    [], chanlocs64, ...
    'style', 'blank', ...
    'electrodes', electrodes, ...
    'plotchans', 1:16, ...
    'emarker', {'.','k',[],0.1}, ...
    'headrad', 'rim', ...
    'conv', conv, ...
    'intrad', intrad, ...
    'intsquare', intsquare ...    
    );

if print_colorbar
    cb = colorbar;
    cb.Ticks = [ limits(1) limits(1)+(limits(2)-limits(1))/2 limits(2) ];
    cb.Label.String = colorbar_string;
end

hold off;

end

