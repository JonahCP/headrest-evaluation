function VecBiosemi64 = myConvChans(VecgTec16)

% this file projects a 16-montage on a 64-montage
% for doing so:
% - the available electrodes are projected 1:1
% - the electrodes neighboring to the 16-montage are scaled
% by a factor wrt their neighbors in the 16-montage
% - all the other electrodes in the 64-montage are set to 0

% within this file, one electrode's neighbors are
% its neighbors in a small laplacian montage

% define 16-montage
Fz =    VecgTec16(1);
FC3 =   VecgTec16(2);
FC1 =   VecgTec16(3);
FCz =   VecgTec16(4);
FC2 =   VecgTec16(5);
FC4 =   VecgTec16(6);
C3 =    VecgTec16(7);
C1 =    VecgTec16(8);
Cz =    VecgTec16(9);
C2 =    VecgTec16(10);
C4 =    VecgTec16(11);
CP3 =   VecgTec16(12);
CP1 =   VecgTec16(13);
CPz =   VecgTec16(14);
CP2 =   VecgTec16(15);
CP4 =   VecgTec16(16);

% vector of zeros
VecBiosemi64 = zeros(1,64);

% scaling factor
factor = 1.25;

% Fz and neighbors
VecBiosemi64(38) = Fz;
% VecBiosemi64(4) = ((Fz/2)+(FC1/2))/factor;  % F1
% VecBiosemi64(37) = Fz/factor;               % AFz
% VecBiosemi64(39) = ((Fz/2)+(FC2/2))/factor; % F2

% FC3 and neighbors
VecBiosemi64(10) = FC3;
% VecBiosemi64(9) = FC3/factor;               % FC5
% VecBiosemi64(5) = FC3/factor;               % F3

% FC1 and neighbors
VecBiosemi64(11) = FC1;
% VecBiosemi64(4)                           % F1, already done in FCz

% FCz and neighbors
VecBiosemi64(47) = FCz;

% FC2 and neighbors
VecBiosemi64(46) = FC2;
% VecBiosemi64(39)                          % F2, already done in FCz

% FC4 and neighbors
VecBiosemi64(45) = FC4;
% VecBiosemi64(40) = FC4/factor;              % F4
% VecBiosemi64(44) = FC4/factor;              % FC6

% C3 and neighbors
VecBiosemi64(13) = C3;
% VecBiosemi64(14) = C3/factor;               % C5

% C1 and neighbors
VecBiosemi64(12) = C1;

% Cz and neighbors
VecBiosemi64(48) = Cz;

% C2 and neighbors
VecBiosemi64(49) = C2;

% C4 and neighbors
VecBiosemi64(50) = C4;
% VecBiosemi64(51) = C4/factor;               % C6

% CP3 and neighbors
VecBiosemi64(18) = CP3;
% VecBiosemi64(17) = CP3/factor;              % CP5
% VecBiosemi64(21) = CP3/factor;              % P3

% CP1 and neighbors
VecBiosemi64(19) = CP1;
% VecBiosemi64(20) = CP1/factor;              % P1

% CPz and neighbors
VecBiosemi64(32) = CPz;
% VecBiosemi64(31) = CPz/factor;              % Pz

% CP2 and neighbors
VecBiosemi64(56) = CP2;
% VecBiosemi64(57) = CP2/factor;              % P2

% CP4 and neighbors
VecBiosemi64(55) = CP4;
% VecBiosemi64(58) = CP4/factor;              % P4
% VecBiosemi64(54) = CP4/factor;              % CP6