function varargout = eegc3_select_gui(varargin)
% eegc3_select_gui M-file for eegc3_select_gui.fig
%      eegc3_select_gui, by itself, creates a new eegc3_select_gui or raises the existing
%      singleton*.
%
%      H = eegc3_select_gui returns the handle to a new eegc3_select_gui or the handle to
%      the existing singleton*.
%
%      eegc3_select_gui('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in eegc3_select_gui.M with the given input arguments.
%
%      eegc3_select_gui('Property','Value',...) creates a new eegc3_select_gui or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before eegc3_select_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to eegc3_select_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help eegc3_select_gui

% Last Modified by GUIDE v2.5 08-Feb-2019 15:05:44

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @eegc3_select_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @eegc3_select_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before eegc3_select_gui is made visible.
function eegc3_select_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to eegc3_select_gui (see VARARGIN)

% Choose default command line output for eegc3_select_gui
handles.output = hObject;
	
% Asses inputs
if(nargin < 4)
    disp('Invalid number of input arguments. Try again!');
end

handles.SessionPlotMats = varargin{1};
handles.DPPlotMat = varargin{2};
handles.SelectedMat = varargin{3};
handles.settings = varargin{4};

set(handles.ModeGroup,'SelectionChangeFcn',@ModeChangeFcn);
% Set mode to automatic
handles.mode = 'auto';
set(handles.AutoBtn,'Value',1);
set(handles.ManBtn,'Value',0);
load('channel_location_16_10-20_mi.mat');
% SessionPlots is a cell array of matrices (DPa mats)
DPLength = length(handles.SessionPlotMats);
for i=1:DPLength
    spax = subplot(1,DPLength,i,'Parent',handles.SessionPlot);
    imagesc(handles.SessionPlotMats{i},'Parent',spax);
    set(gca, 'YTick',      1:handles.settings.acq.channels_eeg);
    set(gca, 'YTickLabel', {chanlocs16.labels});
    set(gca, 'XTick',      [1:1:23]);
    set(gca, 'XTickLabel', num2cell([4:2:48]));
    xlabel('');
    ylabel('');
    plotPresent(handles);
end
waves = {'alpha', 'low beta', 'high beta'};
wavesIndex = [3, 5; 6, 9; 10, 14];
for i=1:DPLength
    for j = 1:length(waves)
        spax = subplot(DPLength,length(waves),(i-1)*length(waves)+j,'Parent',handles.topoplots);
        h = plot_mytopoplot(mean(handles.SessionPlotMats{i}(:,wavesIndex(j,1):wavesIndex(j,2)),2), chanlocs16, 'conv', 'off', 'style', 'map', 'limits', [-1 1], 'electrodes', 'on');
        %title([waves{j} ' run ' num2str(i)]);
    end
end
% Generate the overall DPPlot form the AllDPa matrix
axes(handles.DPPlot);
imagesc(handles.DPPlotMat);
set(handles.DPPlot,'Tag','DPPlot');
plotPresent(handles);

% Set up data cursor
handles.dcm = datacursormode(hObject);
set(handles.dcm,'UpdateFcn',@featureHandler,'Enable','off');

% Set up threshold slider
set(handles.ThSlide,'Value',handles.settings.modules.smr.dp.threshold,'Enable','on');

% Set up selection plot
axes(handles.SelectionPlot);
handles.SelectedHim = imagesc(handles.SelectedMat);
set(handles.SelectionPlot,'Tag','SelectionPlot');
computeSelection(hObject,handles);
   
% Tabs Execution
TabNames = {'SessionPlot','topoplots'};
TabFontSize = 10;
handles = TabsFun(handles,TabFontSize,TabNames);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes eegc3_select_gui wait for user response (see UIRESUME)
uiwait(handles.GUIPanel);

% --- TabsFun creates axes and text objects for tabs
function handles = TabsFun(handles,TabFontSize,TabNames)

% Set the colors indicating a selected/unselected tab
handles.selectedTabColor=get(handles.SessionPlot,'BackgroundColor');
handles.unselectedTabColor=handles.selectedTabColor-0.1;
handles.TabNames = TabNames;

% Create Tabs
TabsNumber = length(TabNames);
handles.TabsNumber = TabsNumber;
TabColor = handles.selectedTabColor;
for i = 1:TabsNumber
    n = num2str(i);
    
    % Get text objects position
    set(handles.(['tab',n,'text']),'Units','normalized')
    pos=get(handles.(['tab',n,'text']),'Position');

    % Create axes with callback function
    handles.(['a',n]) = axes('Units','normalized',...
                    'Box','on',...
                    'XTick',[],...
                    'YTick',[],...
                    'Color',TabColor,...
                    'Position',[pos(1) pos(2) pos(3) pos(4)+0.01],...
                    'Tag',n,...
                    'ButtonDownFcn',[mfilename,'(''ClickOnTab'',gcbo,[],guidata(gcbo))']);
                    
    % Create text with callback function
    handles.(['t',n]) = text('String',TabNames{i},...
                    'Units','normalized',...
                    'Position',[pos(3),pos(2)/2+pos(4)],...
                    'HorizontalAlignment','left',...
                    'VerticalAlignment','middle',...
                    'Margin',0.001,...
                    'FontSize',TabFontSize,...
                    'Backgroundcolor',TabColor,...
                    'Tag',n,...
                    'ButtonDownFcn',[mfilename,'(''ClickOnTab'',gcbo,[],guidata(gcbo))']);

    TabColor = handles.unselectedTabColor;
end
            
% Manage panels (place them in the correct position and manage visibilities)
set(handles.SessionPlot,'Units','normalized')
pan1pos=get(handles.SessionPlot,'Position');
set(handles.tab1text,'Visible','off')
for i = 2:TabsNumber
    n = num2str(i);
    set(handles.(TabNames{i}),'Units','normalized')
    set(handles.(TabNames{i}),'Position',pan1pos)
    set(handles.(TabNames{i}),'Visible','off')
    set(handles.(['tab',n,'text']),'Visible','off')
end

% --- Callback function for clicking on tab
function ClickOnTab(hObject,~,handles)
m = str2double(get(hObject,'Tag'));

for i = 1:handles.TabsNumber
    n = num2str(i);
    if i == m
        set(handles.(['a',n]),'Color',handles.selectedTabColor)
        set(handles.(['t',n]),'BackgroundColor',handles.selectedTabColor)
        set(handles.(handles.TabNames{i}),'Visible','on')
    else
        set(handles.(['a',n]),'Color',handles.unselectedTabColor)
        set(handles.(['t',n]),'BackgroundColor',handles.unselectedTabColor)
        set(handles.([handles.TabNames{i}]),'Visible','off')
    end
end



% Callback to handle the data cursor clicks
function txt = featureHandler(obj,event_obj)

ImageHandle = event_obj.Target;

Pos = get(event_obj,'Position');
txt = {['Frequency bin: ' num2str(Pos(1))], ['Electrode: ' num2str(Pos(2))]};
get(get(ImageHandle,'Parent'),'Tag')
if(strcmp(get(get(ImageHandle,'Parent'),'Tag'),'SelectionPlot')) % Means it is the Selection Plot

    % Only way I found to retrieve the GUI handle in this callback     
    GUIHandle = get(get(ImageHandle,'Parent'),'Parent');
    GUIdata = guidata(GUIHandle);
    
    SelPlotImHandle = findobj(findobj(GUIHandle,'Tag','SelectionPlot'),'Type','image');
    SelectedData = get(SelPlotImHandle,'CData');

    % Remove selection
    SelectedData(Pos(2),Pos(1)) = 0;
    set(SelPlotImHandle,'CData',SelectedData);
    guidata(GUIHandle,GUIdata);
    return;
elseif(strcmp(get(get(ImageHandle,'Parent'),'Tag'),'DPPlot')) % Means it is the DPPlot
 
    % Only way I found to retrieve the GUI handle in this callback     
    GUIHandle = get(get(ImageHandle,'Parent'),'Parent');
    GUIdata = guidata(GUIHandle);
    
    SelPlotImHandle = findobj(findobj(GUIHandle,'Tag','SelectionPlot'),'Type','image');
    SelectedData = get(SelPlotImHandle,'CData');

    % Perform selection
    SelectedData(Pos(2),Pos(1)) = 1;
    set(SelPlotImHandle,'CData',SelectedData);
    guidata(GUIHandle,GUIdata);
    return;
    
else % It is the per session plots
    % Only way I found to retrieve the GUI handle in this callback     
    GUIHandle = get(get(get(ImageHandle,'Parent'),'Parent'),'Parent');
    GUIdata = guidata(GUIHandle);
    
    SelPlotImHandle = findobj(findobj(GUIHandle,'Tag','SelectionPlot'),'Type','image');
    SelectedData = get(SelPlotImHandle,'CData');

    % Perform selection
    SelectedData(Pos(2),Pos(1)) = 1;
    set(SelPlotImHandle,'CData',SelectedData);
    guidata(GUIHandle,GUIdata);
    return;
end


% --- Outputs from this function are returned to the command line.
function varargout = eegc3_select_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure

%Put the final selection in the settings structure, return it and close
% Retrieve data 
SelPlotImHandle = findobj(findobj(hObject,'Tag','SelectionPlot'),'Type','image');
SelectedData = get(SelPlotImHandle,'CData');

handles.settings.bci.smr.channels = [];
handles.settings.bci.smr.bands = cell(1,size(SelectedData,1));
bnidx = cell(1,size(SelectedData,1));
tot = 0;

tot = 0;
for ch=1:size(SelectedData,1)
    for bn = 1:size(SelectedData,2)
        
        if(SelectedData(ch,bn)>0) % if this feature selected...
            
            tot = tot + 1;
            
            if(isempty(find(handles.settings.bci.smr.channels == ch)))
                % Add this channel
                handles.settings.bci.smr.channels = [handles.settings.bci.smr.channels ch];
            end
            
            % Add band to this channel
	    bnidx{ch} = [bnidx{ch} bn];
            handles.settings.bci.smr.bands{ch} = [handles.settings.bci.smr.bands{ch} (2*(bn-1)+4)];
            
            
        end
    end
end

% Set the final value of the threshold to be saved
if(strcmp(handles.mode,'man'))
	handles.settings.modules.smr.dp.threshold = 0; % Denotes manual selection
end
% Otherwise keep the current value, which is the correct one

guidata(hObject,handles);
varargout{1} = handles.settings;
varargout{2} = bnidx;
varargout{3} = tot;
delete(hObject)

% --- Executes on button press in ClassifyBtn.
function ClassifyBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ClassifyBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
uiresume;

% --- Executes on slider movement.
function ThSlide_Callback(hObject, eventdata, handles)
% hObject    handle to ThSlide (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
computeSelection(hObject,handles);
handles.settings.modules.smr.dp.threshold  = get(hObject,'Value');
guidata(get(hObject,'Parent'),handles);

% --- Executes during object creation, after setting all properties.
function ThSlide_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ThSlide (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes during object creation, after setting all properties.
function DPPlot_CreateFcn(hObject, eventdata, handles)
% hObject    handle to DPPlot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate DPPlot


% --- Executes during object creation, after setting all properties.
function SelectionPlot_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SelectionPlot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate SelectionPlot


function ModeChangeFcn(hObject, eventdata)

%retrieve GUI data, i.e. the handles structure
handles = guidata(hObject); 
 
switch get(eventdata.NewValue,'Tag')   % Get Tag of selected object
    case 'AutoBtn'
        handles.mode = 'auto';
        set(handles.dcm,'Enable','off');
        % Get rid of annoying remaining datatips...
        delete(findall(gcf,'Type','hggroup','HandleVisibility','off'));
        
        % Enable threshold slider
        set(handles.ThSlide,'Value',handles.settings.modules.smr.dp.threshold,'Enable','on');
        % Code to compute selection here...
        
        computeSelection(get(hObject,'Parent'),handles);
        
    case 'ManBtn'
        handles.mode = 'man';
        set(handles.dcm,'UpdateFcn',@featureHandler,'Enable','on','DisplayStyle','datatip');
        % Disable slider
        set(handles.ThSlide,'Enable','off');
    otherwise
       % Code for when there is no match.
 
end
%updates the handles structure
guidata(hObject, handles);


function computeSelection(hObject,handles)

Alldpa = reshape(handles.DPPlotMat, size(handles.DPPlotMat,1), ...
	size(handles.DPPlotMat,2));

dpath = (get(handles.ThSlide,'Value')) * max(max(Alldpa));
[cidx, bidx] = eegc3_contribute_matrix(Alldpa, dpath);


handles.SelectedMat = zeros(size(Alldpa));

Allchannels = sort(unique(cidx));
Allbandsidx = {};
for ch = cidx
	Allbandsidx{ch} = [];
end

Alltot = 0;
for i = 1:length(cidx)
	ch = cidx(i);
	bn = bidx(i);
	Allbandsidx{ch} = sort([Allbandsidx{ch} bn]);
	handles.SelectedMat(ch, bn) = Alldpa(ch, bn)/Alldpa(ch, bn);
	Alltot = Alltot + 1;
end


Allbands = {};
for ch = cidx
	Allbands{ch} = ...
	handles.settings.modules.smr.psd.freqs(Allbandsidx{ch});
end

axes(handles.SelectionPlot);
handles.SelectedHim = imagesc(handles.SelectedMat);
set(handles.SelectionPlot,'Tag','SelectionPlot');
plotPresent(handles);
guidata(hObject,handles);


% --- Executes when user attempts to close GUIPanel.
function GUIPanel_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to GUIPanel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
uiresume(hObject);

function plotPresent(handles)
    frequencies = handles.settings.modules.smr.psd.freqs;
    frequencies = num2cell(frequencies);
    for freqIndex = 1:2:length(frequencies)
        frequencies{freqIndex} = '';
    end
    
    load('channel_location_16_10-20_mi.mat');
    set(gca, 'YTick',      1:handles.settings.acq.channels_eeg);
    set(gca, 'YTickLabel', {chanlocs16.labels});
    set(gca, 'XTick',      [1:1:length(handles.settings.modules.smr.psd.freqs)]);
    set(gca, 'XTickLabel', frequencies);
    xlabel('Band [Hz]');
    ylabel('Channel');
    set(findall(gcf,'-property','FontWeight'), 'FontWeight', 'normal');
    set(findall(gcf,'-property','FontSize'),'FontSize',10)
    for row = 1:handles.settings.acq.channels_eeg
        line([0, length(handles.settings.modules.smr.psd.freqs)+1], [row-0.5, row-0.5], 'Color', 'k','LineWidth',1);
    end
    for column = 1 : length(handles.settings.modules.smr.psd.freqs)
        line([column-0.5, column-0.5], [0, handles.settings.acq.channels_eeg+1], 'Color', 'k', 'LineWidth',1);
    end



% --- Executes on button press in tab1text.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to tab1text (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
