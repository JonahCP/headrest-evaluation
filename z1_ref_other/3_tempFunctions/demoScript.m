%% Loading EEG signal

nChannels=64;
nSamples=512;
classes=[1,2];
labels=datasample(classes,200);

load('biosemi64.mat');

% load a random data
a=rand(size(labels,2),nChannels,nSamples);

% Do the laplacian filtering
LaplacianDerivations=f4_eegc3_montage(montage);
save('LaplacianDerivations.mat','LaplacianDerivations');
epochs= f3_LapFilter(a,LaplacianDerivations);

%% Define the pre defined bands

for nBands=1:17
    bands{nBands}=[2*nBands+2,2*nBands+4];
end

features=featExt(epochs,bands);

%% feature selection through fisher

fisherScore=eegc3_fs(features,labels);

[~,indices]=sort(fisherScore,'descend');

SelectedFeatures=features(:,indices);

%% training a classifier

LDA=fitcdiscr(SelectedFeatures,labels);

%% new EEG data


save('bands.mat','bands')
save('LDA.mat','LDA')
save('montage.mat','montage')
newData=rand(nChannels,nSamples);

for n=1:10
    tic;
    filteredData=LapFilter(newData,LaplacianDerivations);

    newFeatures=featExt(filteredData,bands);
    newFeatures=newFeatures(indices);

    predictedLabel=predict(LDA,newFeatures);
    toc
end








