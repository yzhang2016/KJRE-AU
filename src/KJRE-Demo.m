clear; clc ; 


%% Configuration for BP4D
dataName = 'BP4D'; % BP4D
dataIndName = 'BP4D' ; % BP4D
featType = 'NormLMark' ; 
AUInd = [17] ; % [6,10,12,14,17] ; % BP4D 
rateRange = [0.02] ; 


for FN = 100 %:20:120

for ind = 1: length(AUInd)
%% data process
dataPath = sprintf('../data/%s/AU/AUData_lmark_AU%d.mat',dataName,AUInd(ind)) ; 
src = load(dataPath) ; 
seqs = src.seqs; 
cvPath = sprintf('./cvInd_RandEnd_demo/%s_AU_5fds/AU%d',dataIndName,AUInd(ind));
dstPath = sprintf('../Result/RandEnd_demo/%s-Norm/DimEval2/KJREADMM-F%d/AU%d',dataIndName,FN,AUInd(ind)) ; 
if ~exist(dstPath,'dir')
    mkdir(dstPath) ; 
end

%% leave one fd out 
numTimes = 1 ; 

for JJ = 1 : length(rateRange)
annoRate = rateRange(JJ);

TT_unRES = [] ; 
TT_tsRES = [] ; 
TT_rec = []  ; 
cvname = sprintf('%s/AnnoRate_%.2f.mat',cvPath,annoRate) ; 
cvDat = load(cvname); 
cvDat = cvDat.oneRate ; 

if annoRate == 0.0 || annoRate == 1
    TemNumTimes = 1 ; 
else
    TemNumTimes = numTimes ; 
end


for TT = 1 : TemNumTimes 

TTCvDat =  cvDat{TT} ; 

numFds = length(TTCvDat) ; 
tsRES =  zeros(numFds,4) ; 
TsRes = [] ; 
TrAcc =zeros(numFds,4) ;  
unRES = zeros(numFds,4) ; 
temRec = [] ; 

close all ; 
for i = 1 : numFds
    
    SUBDatInd = TTCvDat(i) ; 
    
    % data prepare
    Dat = prepareTrainTestData_AU(seqs,SUBDatInd,'feature',featType) ;   
     
    % % Setting of hyperparameters
    K = FN; 
    NL = size(Dat.YL,1) ; 
    NU = size(Dat.YU,1); 
    baseR = 1/ (NL+NU) ; 
    baseL = 1/ NL ; 
    baseU  =1 /NU ; 

    % BP4D 
    %       |reconstruction |     label     |   L21         | intensity smooth    | feature smooth|
    lambda = [ baseR *1e2 ,     baseL*1e-2,    baseR*1e-3 ,     baseU*1e-4,      baseU*1e-1 ] ; 

    %     | L21         | ordinal   |      Positive|
    rho = [ baseR* 1e0 , baseU*1e-3, baseR*1e-5 ] ;  


    % Learning
    plotflag = 0;
    [w,phiL,phiU,B,fobj,fvec,hisPerf,rec] = ADMM2(Dat.YL,Dat.XL,...
                                               Dat.XUCOM,Dat.LUCOM,Dat.FUCOM,...
                                               K,lambda,rho,...
                                               Dat.numFrmTs, Dat.tsY,...
                                               plotflag);                                   

%====================================================================================  



%% Learning curves
% The meaning of the fvec : ADMM2
Perf = []; 
Perf.fobj = fobj ; 
Perf.fvec = table(fvec(:,1), fvec(:,2),fvec(:,3), fvec(:,4),...
                  fvec(:,5), fvec(:,6), ...
                 'VariableNames',{'reconstruction','label','L21', ...
                                  'L_smooth','F_smooth','Ordinal'});
                              
Perf.hisPerf = hisPerf ; 

temRec = [temRec;Perf] ; 

Z = phiU * w ; 

% Evaluation 
tsZU = Z(Dat.numFrmTr+1:end) ;
tsYU = Dat.YUCOM(Dat.numFrmTr+1:end) ; 

% all unlabeled 
[PCC,UICC,UMAE,UMSE] = OSWMeasure(Dat.YUCOM,Z)  ; 
unRES(i,:) = [PCC, UICC,UMAE,UMSE ] ; 

% for testing sequence
[PCC,UICC,UMAE,UMSE] = OSWMeasure(tsYU,tsZU)  ; 
tsRES(i,:) = [PCC, UICC,UMAE ,UMSE ] ;

plot([tsYU,tsZU]) ; 
    
    
fprintf('Rate = %.2f, Time = %d, SUB = %d,ICC=%.3f, MAE = %.3f ...\n',annoRate,TT,i, UICC, UMAE); 
end
% tsRES(tsRES<0) = 0 ; 
res.XL = Dat.XL; 
res.YL = Dat.YL ; 
res.XUCOM = Dat.XUCOM; 
res.YUCOM = Dat.YUCOM; 
res.phiL = phiL ; 
res.phiU = phiU ; 
res.B = B ; 

save('oswres.mat','res') ; 

avgUNRES = mean(unRES,1) ; 
avgTSRES = mean(tsRES,1) ; 

TT_unRES = [TT_unRES;avgUNRES] ; 
TT_tsRES = [TT_tsRES;avgTSRES] ;
TsRes = [TsRes;{tsRES}] ; 
TT_rec = [TT_rec;{TT_rec}] ; 
end

avgTT_unRES = mean(TT_unRES,1) ; 
stdTT_unRES = std(TT_unRES,1) ; 
avgTT_tsRES = mean(TT_tsRES,1) ; 
stdTT_tsRES = std(TT_tsRES,1) ; 


svname = sprintf('%s/rate_%.2f.mat',dstPath,annoRate) ; 
save(svname,'avgTT_unRES','avgTT_tsRES','stdTT_unRES','stdTT_tsRES','TT_unRES','TT_tsRES','TsRes','TT_rec') ; 
end

end

end
