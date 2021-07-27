function [w,phiL,phiU,B,fobj,fvec,tsPerf,rec,residual] = ADMM2(YL,XL,XU,LU,FU,K,lambda,rho,varargin)
%% Use ADMM to solve the problem
% Feature: combine the L21 norm of unannotated and annotated frames
% @ Created by Yong Zhang, 2017.10.3
%
% Objective: 
%		min 0.5*(lambdaR*|XL-PhiL*B|^2 + 1/NU*|XU-PhiU*B|^2) 
%			+ 0.5*lambda0*(1/NL*|PhiL*w-YL|^2)
%		    + lambda1*(|[PhiL;'PhiU']|_2,1)
%			+ lamdba2*1*w'*PhiU'*LU*PhiU*w
%			+ lambda3*1*tr(PhiU'*LU*PhiU)
%		s.t. FU*PhiU*w <= 0
%			 PhiL*w >= 0, PhiU *w >=0
%			 ||b_i||^2 = 1
%
% Augmented Lagrange function: 
%		min 0.5*(|XL-PhiL*B|^2 + |XU-PhiU*B|^2) 
%			+ 0.5*lambda0*(|PhiL*w-YL|^2)
%		    + lambda1*|C'|_2,1 + 0.5*rho1*|Phi'-C'+A|^2 - 0.5*rho1*|A|^2
%			+ lamdba2*w'*PhiU'*LU*PhiU*w
%			+ lambda3*tr(PhiU'*LU*PhiU)
%			+ I_-(Z0) + 0.5*rho2*1/NU*|FU*PhiU*w-Z0+V0| - 0.5*rho2*1/NU*|V0|^2
%			+ I_+(Z1) + 0.5*rho3*1/NL*|PhiL*w-Z1+V1| - 0.5*rho3*1/NL*|V1|^2
%			+ I_+(Z2) + 0.5*rho3*1/NU*|PhiU*w-Z2+V2| - 0.5*rho3*1/NU*|V2|^2
%
% =================================================
% Input:
% -------------------------------------------------
% 	YL:		The labels of annotated samples 
%	XL:		The features of annotated samples	
% 	XU:		The features of unannotated samples
%	LU: 	The speudo Laplacian matrix for unannotated samples
% 	FU: 	The matrix to compute the ordinal information for unannotated samples
%	K: 		The dimension of the subspace
% 	lambda: A vector of hyperparameters
%			-lambda0: the weight for label loss, |PhiL*w-YL|^2 
%			-lambda1: the weight for L21 norm, |PhiL|_2,1 + |PhiU|_2,1
%			-lambda2: the weight for intensity smoothness, w'*PhiU'*LU*PhiU*w
%			-lambda3: the weight for feature smoothness, tr(PhiU'*LU*PhiU)
%	rho: 	A vector of hyperparameters for ADMM augmented terms
% 			-rho1: the weight for L21 norm, |PhiL-CL|^2 + |PhiU-CU|^2
%			-rho2: the weight for linear constraints, |F*PhiU*w-Z0|^2
%			-rho3: the weight for linear constraints, |PhiL*w-Z1|^2 + |PhiU*w-Z2|^2
%			
% =================================================
% Output:
% -------------------------------------------------
%	w:		The parameters of the regressor in the subspace
%	phiL:	The coefficients of annotated samples 
%	phiU: 	The coefficients of unannotated samples
% 	B:		The  basic vectors of the learned subspace
%	fobj: 	The history of the objective function 
%	fvec: 	The history of each term
%
lambdaR = lambda(1) ;
lambda0 = lambda(2) ; 
lambda1 = lambda(3) ; 
lambda2 = lambda(4) ; 
lambda3 = lambda(5) ; 
rho1 = rho(1) ;
rho2 = rho(2) ;
rho3 = rho(3) ; 

numTs = [] ; 
tsY = [] ; 
if ~isempty(varargin)
    numTs = varargin{1} ; 
    tsY = varargin{2} ; 
    plotflag = varargin{3} ;
end


rec = [] ; % for plotting

%% ======================================= 
% Initialization 
% Two ways for initializaiton: 
%	- 1. PCA
%	- 2. Convex coding AAAI 2011
% PCA
NL = size(XL,1); 
NU = size(XU,1); 


[C,S,~] = pca([XL;XU]); 

S = S(:,1:K);
C = C(:,1:K);  % take the first K components


B = C' ; 
phiL = S(1:NL,:); 
phiU = S(NL+1:end,:); 
CL = phiL ; 
CU = phiU ; 
mul = 0.01 ; 
AL = rand(size(CL')) * mul ; 
AU = rand(size(CU')) * mul ; 

w = rand(K,1);
w = w / norm(w); 
 

Z0 = - rand(NU,1) * mul ; 
Z1 = rand(NL,1) * mul ; 
Z2 = rand(NU,1) * mul ; 

V0 = rand(NU,1) * mul ; 
V1 = rand(NL,1) * mul ; 
V2 = rand(NU,1) * mul ; 

maxIter = 50; 
tolerance = 1e-2;
fobj= [] ; 
fvec = [] ;
tsPerf= [] ;
residual = [] ; 
%% ======================================= 
% Optimization, B, phiL, phiL, w, CL, CU, Z0, Z1, Z2  
LU = sparse(LU);
FU = sparse(FU);
LL = (LU + LU') ; 
FF = FU'*FU ; 

WW = w*w' ;
PLL = phiL' * phiL; 
PUU = phiU' * phiU; 

beta = 1e-8 ; 

%figure sestting 
if plotflag == 1
    hfig = figure ; 
    ylabel('Intensity') ; 
    xlabel('testing frame index');
    title('Performance on the testing data') ; 
    % hold on ; 
end

% constant used in the loop
eyeK = sparse(eye(K)) ; 
eyeNU = sparse(eye(NU)) ; 
T1 = sparse(rho2*FF+rho3*eyeNU); 
T2 = sparse(rho1*eyeNU + 2*lambda3*LU) ; 
T3 = rho3*eyeNU + 2*lambda2*LU + rho2 *FF; 

for iter = 1:maxIter
	%========== Optimize B ==========================================
	B = (PLL + PUU + beta*eyeK) \ (phiL' * XL + phiU' * XU); 
	B = B ./ repmat(sqrt(sum(B.^2,2)), 1, size(B,2)) ; 

    BB = B*B' ; 
	%========== Optimize phiL and phiU =============================
	phiL = (lambdaR*XL*B' + lambda0*YL*w' + rho1*(CL-AL') + rho3*(Z1-V1)*w') ...
			/ (lambdaR*BB +(lambda0+rho3)*WW + rho1*eyeK); 

    % ------ Directly use the gradient to update hte paraemters ----------
 
    tmax = 100 ;
    tfobj= [] ;
    ttol = 1e-4;
    
    T0 = lambda2*WW + lambda3*eyeK; 
    T4 = (rho2*FU'*(Z0-V0)+rho3*(Z2-V2))*w' ; 
    for t =1 :tmax
        
        G = lambdaR*(phiU*BB - XU*B') + rho1*(phiU-CU+AU') ...
              + LL*phiU*T0 + T1*phiU*WW - T4 ; 

        GB = G*B ; 
        GW = G*w ; 
        PW = phiU*w ; 
        PB = phiU*B ; 
        
        tobj = 0.5*lambdaR*sum(sum((XU-PB).^2)) + 0.5*rho1*sum(sum((phiU'-CU'+AU).^2)) ...
            + 0.5*rho2*sum(sum((FU*PW-Z0+V0).^2)) + 0.5*rho3*sum(sum((PW-Z2+V2).^2)) ...
            + lambda2 * PW'*LU*PW + lambda3*trace(phiU'*LU*phiU) ; 
        tfobj = [tfobj;tobj] ; 

        if length(tfobj) > 3 && sum( abs(tfobj(end-3:end)-tfobj(end))) < ttol
            break ;
        end
        
        t1 = - lambdaR*trace((XU-PB)' * GB) - rho1*trace((-phiU+CU-AU')'*G) ...
             - rho2*(-FU*PW+Z0-V0)'*(FU*GW) - rho3*(-PW+Z2-V2)'*GW ... 
             + lambda2*PW'*LL*GW + lambda3* trace(G'*LL*phiU) ; 
         
        t2 = lambdaR*trace(GB'*GB) + trace(G'*T2*G) + GW'*T3*GW; 
        alpha = t1/t2 ; 
%          alpha = 0.1; 
        
        phiU = phiU - alpha * G ; 
    end
    
    PLL = phiL' * phiL; 
    PUU = phiU' * phiU; 
    
	%========== Optimize w ===========================
	t1 = (lambda0+rho3)*PLL + rho3*PUU +  phiU'*(lambda2*LL+rho2*FF)* phiU + beta*eyeK; 
	t2 = lambda0*phiL'*YL + rho2*phiU'*FU'*(Z0-V0) + rho3*phiL'*(Z1-V1) + rho3*phiU'*(Z2-V2) ; 
	w = t1\t2;

	%========== Optimize CL and CU ===================
    % together 
    C = zeros(NL+NU,K) ; 
    A = [AL,AU] ; 
    P = [phiL;phiU] ; 
    for i = 1 : K
        C(:,i) = softThresh(lambda1/rho1, P(:,i) + A(i,:)') ; 
    end
    CL = C(1:NL,:) ; 
    CU = C(NL+1:end,:) ; 
    
    
	%========== Optimize Z0 and Z1,Z2 ================ADMM.m
	Z0 = min(0,FU*phiU*w + V0);
	Z1 = max(0,phiL*w + V1);
	Z2 = max(0,phiU*w + V2);

	%========== Optimize AL,AU,V0,V1,V2===============
	AL = AL + (phiL' - CL');
	AU = AU + (phiU' - CU'); 
	V0 = V0 + (FU*phiU*w - Z0); 
	V1 = V1 + (phiL*w - Z1); 
	V2 = V2 + (phiU*w - Z2); 

    
    %============compute residual ===================
    res_AL = sqrt(sum(sum((phiL-CL).^2))) ; 
    res_AU = sqrt(sum(sum((phiU - CU).^2))) ; 
    res_V0 = sqrt(sum(sum((FU*phiU*w - Z0).^2))) ; 
    res_V1 = sqrt(sum(sum((phiL*w - Z1).^2))) ;
    res_V2 = sqrt(sum(sum((phiU*w - Z2).^2))) ;
    
    residual = [residual;res_AL,res_AU, res_V0,res_V1,res_V2] ; 
    
    
    
    
    
	[tfobj,tfvec] = objective(YL,XL,XU,LU,FU,K,lambda,rho,...
						phiL,phiU,B,w,CL,CU,AL,AU,Z0,Z1,Z2,V0,V1,V2); 
	fobj = [fobj;tfobj] ; 
	fvec = [fvec;tfvec] ; 
	
% 	if length(fobj) > 6 && ( sum(abs(fobj(end-3:end) - fobj(end))) < tolerance || fobj(end-1) < fobj(end) )
% 		break; 
%     end	
    
    % ------------------------
    % test 
    if ~isempty(numTs) 
        preTsY =  phiU(end-numTs+1:end,:) * w ; 
        [PCC,UICC,UMAE,UMSE] = OSWMeasure(tsY,preTsY)  ;
        tsPerf = [tsPerf;PCC,UICC,UMAE,UMSE] ; 
        if iter > 20 && tsPerf(end,2) <  tsPerf(end-1,2) 
            break ; 
        end	 
        if plotflag == 1
            plot(tsY,'-b'); 
            plot(preTsY,'-r') ;  
            refresh(hfig)  ; 
            pause(0.5) ; 
            
            rec = [rec;{[tsY,preTsY]}] ; 
            
        end
        fprintf('Iter = %d, objective = %.4f, ICC = %.3f, MAE = %.3f ... \n',iter,tfobj(end), UICC, UMAE) ; 
    else 
        fprintf('Iter = %d, objective = %.4f ... \n',iter,tfobj(end)) ; 
        if length(fobj) > 6 && ( sum(abs(fobj(end-3:end) - fobj(end))) < tolerance )
            break; 
        end	
    end
    
%     fprintf('Iter = %d, objective = %.4f... \n',iter,tfobj(end)) ; 
end

function [fobj,fvec] = objective(YL,XL,XU,LU,FU,K,lambda,rho,...
							phiL,phiU,B,w,CL,CU,AL,AU,Z0,Z1,Z2,V0,V1,V2)
%% Compute the value of the objective function
 
lambdaR = lambda(1) ;
lambda0 = lambda(2) ; 
lambda1 = lambda(3) ; 
lambda2 = lambda(4) ; 
lambda3 = lambda(5) ; 
rho1 = rho(1) ;
rho2 = rho(2) ;
rho3 = rho(3) ; 


% reconstruction loss
t0 = 0.5*lambdaR*(sum(sum((XL-phiL*B).^2)) + sum(sum((XU-phiU*B).^2))); 

% supervision loss
t1 = 0.5*lambda0*sum(sum((phiL*w-YL).^2)); 

% L21 norm
t2 = lambda1 * L21Norm([CL;CU]') + 0.5*rho1*sum(sum(([phiL;phiU]'-[CL;CU]'+ [AL,AU]).^2)) - 0.5*rho1*sum(sum(([AL,AU]).^2)); 

% intensity smoothness
t3 = lambda2*w'*phiU'*LU*phiU*w ; 

% feature smoothness
t4 = lambda3 * trace(phiU'*LU*phiU); 

% ordinal intensity
t5 = (0.5*rho2*sum(sum((FU*phiU*w-Z0+V0).^2)) - 0.5*rho2*sum(sum(V0.^2))) ; 

% intenisty larger than 0
t6 =   0.5*rho3*sum(sum((phiL*w-Z1+V1).^2)) - 0.5*rho3*sum(sum(V1.^2)) ...
     + 0.5*rho3*sum(sum((phiU*w-Z2+V2).^2)) - 0.5*rho3*sum(sum(V2.^2)) ; 

fvec = [t0 , t1 , t2 , t3 , t4 , t5 , t6] ;
fobj = sum(fvec);

function [L21] = L21Norm(A)
%% Compute the L21 norm of A
L21 = 0 ; 
for i =1:size(A,1)
	L21 = L21 + norm(A(i,:)) ; 
end


function [sa] = softThresh(k,a)
%% 
% k: the thresh 
% a: the input vector 
if sum(a.^2) == 0
	sa = a ; 
else
	sa = max(0,1-k/norm(a)) * a ; 
end