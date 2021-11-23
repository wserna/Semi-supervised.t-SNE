function Y = SStSNE(X,L,dim,thresh_cat,itr,tol,mso,fpt)
% Function
% 
% Y = SStSNE(X,L,dim,thresh_cat,itr,tol,fpt)
% 
% performs similarity-based dimensionality reduction,
% The similarities are single-scale for labeled data and 
% multiscale for unlabeled data.
% 
% Principal component are used to initialize embedding Y.
% The optimization involves a line search with backtracking for the step
% size adjustment under the strong Wolfe conditions. The search direction 
% is the product of the gradient with a diagonal estimate of the Hessian, 
% which is refined by the limited-memory BFGS algorithm (m=7).
% The optimization is multiscale, in the sense that BFGS is run several 
% times, starting with the largest perplexity value and then introducing
% or switching to smaller ones.
% 
% Inputs:
%   X  : n*m matrix where each row is a point in HD space with m variables
%   L  : n*1 matrix where L(i)is the class of sample X(i,:)
%        if L(i) == -1, X(i,:)is an unlabeled point.
%   dim : embedding dimensionality (scalar integer; 2)
%   thresh_cat : ([0.5,1[) Percentage of points of the same class in
%                the soft neighborhood of a labeled point.
%   itr : maximum number of iterations per perplexity value
%   tol : tolerance for stopping criterion (tiny scalar float; 1e-4)
%   mso : true or false for Multi-scale optimization.
%   fpt : specifies the floating-point type (scalar; 2)
%         positive => CPU, negative => GPU (no guarantee for GPU!)
%         abs(fpt)<2 => single precision, double otherwise
%
% Outputs:
%   Y   : coordinates in the LD space of the final embedding
%
% References:
%
% [1] John A. Lee
%     Type 1 and 2 mixtures of divergences for stochastic neighbor embedding
%     Proc. ESANN 2012, 20:525-530.
% [2] J. A. Lee, E. Renard, G. Bernard, P. Dupont, M. Verleysen
%     Type 1 and 2 mixtures of Kullback-Leibler divergences
%     as cost functions in dimensionality reduction
%     based on similarity preservation
%     Neurocomputing 2013, 112: 92-108.
% [3] John A. Lee, Diego H. Peluffo, Michel Verleysen
%     Multi-scale similarities in stochastic neighbour embedding:
%     Reducing dimensionality while preserving both local and global structure
%     Neurocomputing 2015, 169:246-261.
%     http://dx.doi.org/10.1016/j.neucom.2014.12.095
% [4] de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & 
%     Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414).

%% process arguments ------------------------------------------------------

% default arguments
if nargin<8, fpt = 2; end
if nargin<7, mso = true; end % multiscale optimisation
if nargin<6, tol = 1e-4; end
if nargin<5, itr = 1000; end
if nargin<4, thresh_cat = 0.9; end
if nargin<3, dim = 2; end
if nargin<2, error('Insufficient number of arguments!'); end

disp('Converting coordinates into pairwise distances')
DX = pdist2(X,X); %euclidean distance among samples
disp('Initializing with principal components');
[~,Y] = pca(X,'NumComponents',dim); %Initializing LD coordinates
n = size(DX,1);

Un_Lidx = L == -1; %Unlabeled point index
Lb_Lidx = L ~= -1; %Labeled point index

% floating-point type and GPU computation
try
    gpu = (fpt<0) && (1<=gpuDeviceCount);
catch
    gpu = false;
end
if abs(fpt)<2
    fpt = 'single';
else
    fpt = 'double';
end

% convert to specified floating-point type
DX = feval(fpt,DX);
Y = feval(fpt,Y);

% transfer to GPU if requested
if gpu
    DX = gpuArray(DX);
    Y = gpuArray(Y);
end

%% precisions in the HD space ---------------------------------------------

% majorize and exponentiate distances in the HD space
mDXp = 1/2*max(DX,0).^2;

%% multiscale neighborhood radii
disp('Soft neighborhood equalization:');

npl = round(log2(n/2)); % number of perplexity levels (up to nbr/2)
ens = 2*2.^(npl-1:-1:0);

% compute precisions in HD space
PX = cell(npl,1);
PXi = [];
for i = 1:npl
    % entropy equalization
    PXi = equalize(mDXp,PXi,ens(i),itr);
    PX{i} = PXi;
    
    if i>2
        PXi = PXi.*(PX{i-1}./PX{i-2}); % guess based on previous ratio => spare at least one iteration
    end
end

%Clase aware precisions

    LL = L*ones(1,n);
    LL = bsxfun(@eq, L', LL);
    CX = zeros(npl,n);
    caidx = npl*ones(1,n);
    offset = min(mDXp(DX>eps));
    caPX = zeros(1,n); % precisions
    caMX = zeros(1,n); % max. class representation
    Eyeneg = (ones(n)-eye(n));
    for scl = npl:-1:1
        P = exp(bsxfun(@times, -PX{scl}, mDXp - offset.*Eyeneg));
        P = P - eye(n);
        for i = 1:n
            CX(scl,i) = sum(P(LL(:,i),i),1)/ (sum(P(:,i),1));
            if CX(scl,i)>=thresh_cat*(1-sum(P(Un_Lidx,i),1)/ (sum(P(:,i),1)))
                caidx(i) = scl;
            elseif scl<npl && CX(scl,i)>CX(scl+1,i)
                caidx(i) = scl;
            end
            caPX(i) = PX{caidx(i)}(i);
            caMX(i) = CX(caidx(i),i);
        end
    end
    
    caidx(Un_Lidx) = npl; %Always update unlabeled data

%% main part: iterate until convergence -----------------------------------
disp('Cost function minimization:')

% embedding optimization

stp = 1;
if mso, fpl = 1; else fpl = npl; end % first perplexity level => number of optimization levels
for cpl = fpl:npl
    disp(['thresh_cat ', num2str(thresh_cat), ', Level ',num2str(cpl),'/',num2str(npl),': perplexity ',num2str(ens(cpl)),' to ',num2str(ens(1))]);

    % perplexity index range
    pir = 1:cpl;

    % compute HD similarities
    [nSX,lnSX] = multisimilarities(mDXp,PX(pir),Lb_Lidx,caidx);

    % first function evaluation
    F = zeros(1,itr,'like',Y); % records of all objective function values
    ncfe = zeros(1,itr); % number of cost function evaluation
    px = Y(:);

    [pf,pg,ph] = cfe(px,nSX,lnSX,dim); % first cost function evaluation

    ncfe(1) = 1;
    
    % diagonal approximation of Hessian (regularized)
    dH = mean(abs(ph));
    Hg = pg./dH; % first search direction
    
        % limited memory BFGS
        
        % lmBFGS variables
        hlm = 7; % history length
        xdifs = zeros(numel(px),hlm,'like',px);
        gdifs = zeros(numel(px),hlm,'like',px);
        
        % Wolfe conditions and backtracking variables
        b1 = 1e-2; % Wolfe: tiny gains are allowed
        b2 = 0.9; % Wolfe
        nst = 10; % maximum number of steps
        ys = zeros(nst,1,'like',pf);
        xs = zeros(nst,1,'like',pf);
        
        % iterate
        tbis = 0;
        for t = 1:itr
            % record current cost function value
            F(t) = pf;
            
            % (loose) line search with backtracking
            ostp = stp;
            for i = 1:nst
                % new x (descent along search direction)
                x = px - stp*(Hg-mean(Hg)); % remove mean to avoid translational freedom
                
                % cost function evaluation
                [f,g,h] = cfe(x,nSX,lnSX,dim);

                ncfe(t) = ncfe(t) + 1;
                
                % record
                ys(i) = f - pf;
                xs(i) = stp;
                
                % Wolfe conditions (Armijo + strong condition on curvature)
                tmp = pg'*Hg;
                if f<=pf+stp*b1*tmp && abs(g'*Hg)<=abs(b2*tmp)
                    stp = 1.25*stp; % confident for next search ;-)
                    break; % stop on first success
                else
                    % oops! went to far... continue cautiously
                    if i<2
                        % dichotomic decrease
                        stp = 0.5*stp;
                    elseif ys(i)/ys(i-1)<(xs(i)/xs(i-1))^2
                        % quadratic interpolation
                        qa = ys(i)*xs(i-1)         - ys(i-1)*xs(i)      ;
                        qb = ys(i)*xs(i-1)*xs(i-1) - ys(i-1)*xs(i)*xs(i);
                        stp = max(0.125*stp, qb/(2*qa) ); % between 1/8 and 1/2
                    else
                        % faster decrease
                        stp = 0.125*stp;
                    end
                end
            end
            
            % workarounds
            if i==nst
                disp(['l-BFGS reinitialization at it. ',num2str(t),' (unsuccessful line search)']);
                tbis = 1;
                stp = min(1,2*ostp);
            else
                tbis = tbis + 1;
            end
            
            % stopping criterion
            if t>=5 && abs(1 - F(t)/F(t-4))<tol
                disp(['Stop at it. ',num2str(t),' (tolerance reached)']);
                break;
            end
            
            % determine Hessian approximation for next line search
            
            % positive definite diagonal approximation of Hessian (regularized)
            dH = mean(abs(h));
            % the Hessian is rank-deficient and prod(h) can be null!!!
            % (due to plasticity: gradient and curvature are null in some directions
            % if outliers are present; this issue is not corrected by lmBFGS
            
            % approximation of the inverse Hessian with lmBFGS "double recursion"
            xdifs = [x-px,xdifs(:,1:hlm-1)];
            gdifs = [g-pg,gdifs(:,1:hlm-1)];
            r = 1./(sum(gdifs.*xdifs,1));
            r = min(max(r,-1/epsmax),1/epsmax); % avoid too large coefficients
            a = zeros(1,hlm,'like',x);
            b = zeros(1,hlm,'like',x);
            q = g;
            for i = 1:+1:min(t,hlm) % backward loop (if hlm>0)
                a(i) = r(i)*xdifs(:,i)'*q;
                q = q - a(i)*gdifs(:,i);
            end
            Hg = q./dH; % search direction
            for i = min(t,hlm):-1:1 % forward loop (if hlm>0)
                b(i) = r(i)*gdifs(:,i)'*Hg;
                Hg = Hg + (a(i)-b(i))*xdifs(:,i);
            end
            % (hlm==0) => (Hg==g)
            
            % updates
            pf = f;
            px = x;
            pg = g;
        end
    
    % rebuild coordinates from generic unknown vector x
     Y = reshape(x,n,dim);
end

%% finalize outputs -------------------------------------------------------

% PCA decorrelation (to avoid translational and rotational freedom)
Y = bsxfun(@minus, Y, sum(Y,1)./n); % mean subtraction
[U,S] = svd(Y,0); % SVD PCA on coordinates
Y = bringback(U * S); % back from GPU to CPU and double precision after rotation

%% additional functions

function [f,g,h] = cfe(x,nSX,lnSX,dim)
% cost function evaluation (with gradient and diagonal of Hessian)

% sizes
n = size(nSX,1);
sss = size(nSX,2);
ss = 1:sss;

% reconstruct output coordinates from generic BFGS vector x
Y = reshape(x,n,dim);

% half majorized squared Euclidean distances in the LD space
DY2 = Y * Y';
DY2 = bsxfun(@minus,diag(DY2),DY2);
DY2 = DY2 + DY2';
mDYp = max(0,DY2); % much faster on GPU    

% compute LD similarities
[nSY,lnSY, g1, g2] = similaritiesTSNE(mDYp);

%Kullback-Leibler
bd0 = -nSX.*lnSX; 
bpcd0 = nSY-nSX + nSX.*(lnSX-lnSY);
u1cd1 = nSY-nSX;
u2cd2 = nSX;

% non-negligible entries
nni = (0<mDYp);

% cancel all discarded terms and multiply by weights
W0 = sum(nni,1)./sum(abs(bd0.*nni),1);
bpcd0 = bsxfun(@times, W0, bpcd0.*nni);
u1cd1 = bsxfun(@times, W0, u1cd1.*nni);
u2cd2 = bsxfun(@times, W0, u2cd2.*nni);

% compute cost function value
f = sum(sum(bpcd0,1),2);

% compute gradient and Hessian weights
W1 = sum(u1cd1(:))/n*nSY - u1cd1;
W2 = (2*nSY-1).*(W1 - u2cd2) + sum(u2cd2(:))/nbr*nSY.^2;

% symmetrization
W1 = 0.5*(W1 + W1');
W2 = 0.5*(W2 + W2');

% compute the gradient and approximate diagonal Hessian
Ypd1 = zeros(size(Y),'like',Y); % gradient w.r.t. Y
Ypd2 =  ones(size(Y),'like',Y); % diag. Hessian w.r.t. Y (crude approximation)
tmp1 = W1.*g1;
tmp2 = W1.*g2 + W2.*g1.^2 ;
tmp1ss = tmp1(ss,ss);
tmp1(ss,ss) = 0.5*(tmp1ss + tmp1ss'); % "late" symmetrization for "marginal" (symmetric anyway for "joint")
tmp2ss = tmp2(ss,ss);
tmp2(ss,ss) = 0.5*(tmp2ss + tmp2ss'); % "late" symmetrization for "marginal" (symmetric anyway for "joint")

for d = 1:dim
    Yd = Y(:,d);

    dif = bsxfun(@minus, Yd, Yd'); % much faster on GPU

    Ypd1(:,d) = sum( tmp1.*dif              ,2);
    Ypd2(:,d) = sum( tmp1      + tmp2.*dif.^2 ,2);
end

% reshape for BFGS
g = Ypd1(:);
h = Ypd2(:);

function x = bringback(x)
try
    % try to transfer from GPU to CPU
    x = double(gather(x));
catch
    % nothing to do
end

function lgmn = logmin(x)
% compute the minimal logarithm value for the floating-point precision of x

% class of variable x
if isa(x,'gpuArray')
    fpt = classUnderlying(x);
else
    fpt = class(x);
end

% depending on the floating-point class
switch fpt
    case 'single', lgmn = feval(fpt,8e-46);  % SP: log(8e-46) = -103.2789
    case 'double', lgmn = feval(fpt,8e-46);  % DP: mimic behavior of SP
end

function epmx = epsmax
% largest epsilon
epmx = 2^(-23); % this is equivalent to eps(single(1.0)), without forcing the class to be 'single'

function [nSX,lnSX] = multisimilarities(DX,PX,Lb_Lidx,caidx)
% multiscale similarities
% DX: matrix of half squared distances
% PX: cell of precisions vectors (one row vector per level)
% W : weights (for of each scale)
% nSX: normalized similarities
% lnSX: logarithm of normalized similarities
% (gpu-compatible)

% number of levels
npl = size(PX,1);
N = size(DX,1);

if npl<=1
    % reduce to single-scale similarities (log.sim. more accurate)
    [nSX,lnSX] = similarities(DX,PX{1});
%     nSX(L~=0,L==0)=0;
else
    % for all scales
    W = ones(N,npl,'like',DX);
    for i = 1:npl
        W(logical(Lb_Lidx.*(i~=caidx')),i) = 0;
        if(i==npl)
            W(npl<caidx,i)=1;
        end
    end
    W = W./repmat(sum(W,2),1,npl);
    nSX = zeros(N,N,'like',DX);
    for i = 1:npl
        nSX = nSX + bsxfun(@times,W(:,i),similarities(DX,PX{i}));
    end
    
    % log(similarities)
    if nargout>1
        lnSX = log(max(logmin(DX),nSX));
    end
end

function [nSX,lnSX] = similarities(DX,PX)
% similarities
% DX: matrix of half squared distances
% PX: row vector of precisions
% nSX: normalized similarities
% lnSX: logarithm of normalized similarities
% (gpu-compatible)

% zero and negligible entries
zni = (DX<=1e-64);

% exploit shift invariance to improve num. accuracy
tmp = DX;
tmp(zni) = inf; % much slower with NaN, inf not slower than any regular value
DX = bsxfun(@minus, DX, min(tmp,[],1)); % subtract minimum distance

% exponential/Gaussian similarity
lSX = bsxfun(@times, -PX, DX);
lSX(zni) = -746; % smallest integer n such that exp(n)==0 in double precision (simple precision too)
SX = exp(lSX);

% normalize similarities
sSX = sum(SX,1); % "marginal" normalization factor
nSX = bsxfun(@rdivide, SX, sSX);

% log(similarities)
if nargout>1, 
    lnSX = bsxfun(@minus, lSX, log(sSX));
end

function [nSY,lnSY, g1, g2] = similaritiesTSNE(DX)
% similarities with Student T
% DX: matrix of half squared distances
% nSX: normalized similarities
% lnSX: logarithm of normalized similarities
N=size(DX,1);
lSY = bsxfun(@times, -1, log1p(DX));
SY = exp(lSY);
SY(1:N+1:end) = 0;
sSY = (sum(SY(:)))/N;% * ones(1,N); % "joint" normalization factor
nSY = SY/sSY;
g1 = 2./(1+DX);
g2 = bsxfun(@times, -1, g1.*g1);
% log(similarities)
lnSY = bsxfun(@minus, lSY, log(sSY));

function [PX,nSX,lnSX] = equalize(mDXp,PX,pxt,itr)
% baseline equalization
% (gpu-compatible)

% initialize precisions
if isempty(PX)
    tmp = mDXp;
    tmp(tmp<=1e-64) = inf; % much slower with NaN, inf not slower than any regular value
    tmp = mean(mDXp,1) - min(tmp,[],1); % shift invariance
    PX = 1./tmp;
end

% Shannon entropy (pxt is the perplexity)
Htgt = log(abs(pxt));

disp(['Target perplexity = ',num2str(pxt)]);

% equalization with Newton's method
% (precisions are initialized above)
for t = 1:itr
    % compute normalized similarities in the HD space
    [nSX,lnSX] = similarities(mDXp,PX);
    
    % compute all entropies
    Htmp = -sum(nSX.*lnSX,1);

    % delta H
    Htmp = Htmp - Htgt;

    % stop or update
    if all(abs(Htmp)<1e-3*abs(Htgt)), break; end

    % update
    PX = PX - max(-PX/2, min(PX/2, Htmp./sum(nSX.*(1+lnSX).*bsxfun(@minus,mDXp,sum(nSX.*mDXp,1)),1) ) ); % allow PX to be multiplied by min. 0.5 or max. 1.5
end
