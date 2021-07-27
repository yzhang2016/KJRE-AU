function [normFeat] = featNormalize(feat,maxVal,minVal)
%% normalize the features 
% maxVal: a vector, the maximum of each dimension 
% minVal: a vector , teh minimum of each dimension 

numSamp = size(feat,2) ; 
normFeat = (feat - repmat(minVal,1,numSamp) )./ repmat(maxVal-minVal,1,numSamp) ; 