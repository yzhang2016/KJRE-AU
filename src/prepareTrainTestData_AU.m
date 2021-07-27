function [Dat] = prepareTrainTestData_AU(seqs,DatInd,varargin)
%% Prepare training and testing for AU recognition 

featStr = 'LMark' ; 
for i =1 :2: length(varargin)
    if strcmp(varargin{i},'feature')
        featStr = varargin{i+1} ; 
    else
        error('Not such a key...') ; 
    end
end

Feat = [] ; 
AUINT = [] ; 
SUBInd = [] ; 

for i = 1:length(seqs)
    tem = seqs(i) ; 
    if strcmp(featStr,'NormLBP')
        Feat = [Feat;{tem.LBPFeat}] ;  % LBP feature (97% energy)
    elseif strcmp(featStr,'PCANormLMark')
        Feat = [Feat;{tem.LMark_PCA}] ; % PCA landmark fetaure (97% energy)
    elseif strcmp(featStr,'NormLMark')
        Feat = [Feat;{tem.LMark_Norm}] ; % nomarlized landmark fetaure 
    else
        error('Wrong: not such feature') ; 
    end

    AUINT = [AUINT; {tem.AUINT}] ; 
    SUBInd = [SUBInd;tem.SUBInd] ; 
end

trainInd = DatInd.trainInd  ; 
testInd = DatInd.testInd ; 

trainFeat = [] ; 
trainINT = [] ; 
testFeat = [] ; 
testINT = [] ; 

numTrSeq  = length(trainInd) ; 
numTsSeq = length(testInd) ; 

%% testing 
for  i = 1 : numTsSeq
    temInd = testInd(i) ; 
    temFeat = Feat{temInd}'; 
    tem = AUINT{temInd}(:,1)' ; 
    temINT = AUINT{temInd}(:,2)';

    if tem(1) > tem(end)
        temFeat = temFeat(:,end:-1:1) ; 
        temINT = temINT(end:-1:1) ; 
    end
    
    testFeat = [testFeat,temFeat] ; 
    testINT = [testINT,temINT] ; 
end

tsX = testFeat; 
tsY = testINT ; 

sL = [] ; 
sF = [] ; 
numFrmsRec = [] ; 
isPSD = 0 ;

%% training 
for  i = 1 : numTrSeq
    temInd = trainInd(i) ; 
    temFeat = Feat{temInd}'; 
    temINT = AUINT{temInd}(:,2)' ; 
    
    trainFeat = [trainFeat,temFeat] ; 
    trainINT = [trainINT,temINT] ; 
    
    % compute LU: D - C
    numFrames = size(temFeat,2) ; 
    numFrmsRec = [numFrmsRec,numFrames] ; 
    C0 = triu(ones(numFrames),-1) - triu(ones(numFrames),2) ;
    D0 = diag(sum(C0,2)) ; 
    L0 = (D0 - C0) ;
    
    if isPSD
%         L0 = nearestSPD(L0) ; 
        if isempty(L0)
            q = 1 ; 
        end
        L0 = Spd_Mat(L0); 
    end
    
    sL = [sL,{L0}] ; 
    
    % compute FU: 
    F0 = zeros(numFrames) ; 
    for p =1 : numFrames - 1 
        F0(p,p) = 1 ; 
        F0(p,p+1) = -1 ; 
    end
    sF = [sF,{F0}] ;     
end

XU = trainFeat ; 
YU = trainINT ; 

numFrmTr = sum(numFrmsRec) ; 
numFrmTs = length(tsY) ; 

% check 
if numFrmTr ~= DatInd.numFrmTr || numFrmTs ~= DatInd.numFrmTs
    error('Data index not compatible...') ; 
end

wholeFrm = numFrmTr +  numFrmTs; 

LUCOM = zeros(wholeFrm) ; 
FUCOM = zeros(wholeFrm) ; 

temSum = 0 ; 
for i =1 : length(numFrmsRec)
    sInd = temSum + 1 ; 
    eInd = sInd + numFrmsRec(i)-1 ; 
    
    LUCOM(sInd:eInd,sInd:eInd) = sL{i} ; 
    FUCOM(sInd:eInd,sInd:eInd) = sF{i} ; 
    temSum = temSum + numFrmsRec(i) ; 
end


% select annotated frames
selInd = DatInd.selInd ; 
XL = XU(:,selInd) ; 
YL = YU(:,selInd) ; 

XUCOM = [XU,tsX] ; 
YUCOM = [YU,tsY] ;

% For transductive learning 
Dat.XL = XL' ; 
Dat.YL = YL' ; 
Dat.XUCOM = XUCOM' ; 
Dat.YUCOM = YUCOM' ; 
Dat.LUCOM = LUCOM ; 
Dat.FUCOM = FUCOM ; 
Dat.numFrmTr = numFrmTr ; 
Dat.numFrmTs = numFrmTs ; 

% For inductive learning 
Dat.XU = XU' ; 
Dat.YU = YU' ; 
Dat.tsX = tsX' ; 
Dat.tsY = tsY'; 
Dat.LU = LUCOM(1:numFrmTr,1:numFrmTr) ; 
Dat.FU = FUCOM(1:numFrmTr,1:numFrmTr) ; 

