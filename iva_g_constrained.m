function [W,cost,Sigma_N] = iva_g_constrained(X,guess_mat,constraint_type,varargin)
% By default this code will perform constraint IVA-G. For regular IVA-G set
% 'constraint_flag' to 0. 
%
% Implementations of all the second-order (Gaussian) independent vector
% analysis (IVA) algorithms.  Namely real-valued and complex-valued with
% circular and non-circular using gradient, Newton, and quasi-Newton
% optimizations.
%
% Input:
%   X - data observations from K data sets, i.e. X{k}=A{k}*S{k}, where A{k}
%       is an N x N unknowned invertible mixing matrix and S{k} is N x T  matrix
%       with the nth row corresponding to T samples of the nth source in the kth
%       dataset.  For IVA it is assumed that the source is statistically
%       independent of all the sources within a dataset and exactly dependent on
%       at most one source in each of the other datasets. The data, X, can be
%       either a cell array as indicated or a 3-dimensional matrix of dimensions
%       N x T x K.  The latter enforces the assumption of an equal number of
%       samples in each dataset.
%   guess_mat - matrix where each column is a reference vector. num_guess
%       is the number of reference signals.
%   constraint_type - set 0 for constraining column(s) of mixing matrix, 1 for constrain source(s)
%       If 0 guess_mat is a N x num_guess matrix. If 1 guess_mat is a T x num_guess matrix
% Output:
%   W - the estimated demixing matrices so that ideally W{k}*A{k} = P*D{k}
%       where P is any arbitrary permutation matrix and D{k} is any diagonal
%       invertible (scaling) matrix.  Note P is common to all datasets; this is
%       to indicate that the local permuation ambiguity between dependent sources
%       across datasets should ideally be resolved by IVA.
%
%   cost - the cost for each iteration
%
%   isi - joint inter-symbol-interference is available if user supplies true
%       mixing matrices for computing a performance metric
%
% Optional input pairs and default values are given in list below:
%    
%    'mu',0.5, ... % initial step parameter for constrained algorithm update (default 0.5)
%    'rho', 0.7, ... % correlation threshold for constrained algorithm (default 0.7)
%    'gam',3, ... % lagrange multiplier step parameter (default 3)
%
%    'opt_approach','gradient', ... % optimization type: gradient, (newton), quasi
%    'complex_valued',false, ... % if any input is complex or quasi approach used then setting is forced to true
%    'circular',false, ... % set to true to only consider circular for complex-valued cost function
%    'whiten',true, ... % whitening is optional (except for quasi approach it is required)
%    'verbose',false, ... % verbose true enables print statements
%    'A',[], ... % true mixing matrices A, automatically sets verbose
%    'W_init',[], ... % initial estimates for demixing matrices in W
%    'jdiag_initW',false, ... % use CCA (K=2) or joint diagonalization (K>2)
%    'maxIter',512, ... % max number of iterations
%    'WDiffStop',1e-6, ... % stopping criterion
%    'alpha0',1.0 ... % initial step size scaling (will be doubled for complex-valued)
%    
% Example call:
% W=iva_g_constrained(X,guess_mat,0,'rho',0.8)
%
% Coded by Matthew Anderson (matt.anderson at umbc.edu)
% Code for constrain IVA added by Suchita Bhinge (suchita1 at umbc.edu)
%
% References:
%
% [1] M. Anderson, X.-L. Li, & T. Adali, "Nonorthogonal Independent Vector Analysis Using Multivariate Gaussian Model," LNCS: Independent Component Analysis and Blind Signal Separation, Latent Variable Analysis and Signal Separation, Springer Berlin / Heidelberg, 2010, 6365, 354-361
% [2] M. Anderson, T. Adali, & X.-L. Li, "Joint Blind Source Separation of Multivariate Gaussian Sources: Algorithms and Performance Analysis," IEEE Trans. Signal Process., 2012, 60, 1672-1683
% [3] M. Anderson, X.-L. Li, & T. Adali, "Complex-valued Independent Vector Analysis: Application to Multivariate Gaussian Model," Signal Process., 2012, 1821-1831
% [4] S. Bhinge, 
%
% Version 01 - 20120913 - Initial publication
% Version 02 - 20120919 - Using decouple_trick function, bss_isi, whiten, &
%                         cca subfunctions
% Version 03 - 20160425 - constrain IVA, ...... added by Suchita Bhinge
if nargin==0
   help iva_g_constrained
   test_iva_g_constrained
   return
end


%% Gather Options

% build default options structure
options=struct( ...
   'opt_approach',1, ... % optimization type: gradient, (newton), quasi
   'whiten',true, ... % whitening is optional (except for quasi approach it is required)
   'verbose',false, ... % verbose true enables print statements
   'A',[], ... % true mixing matrices A, automatically sets verbose
   'W_init',[], ... % initial estimates for demixing matrices in W
   'jdiag_initW',false, ... % use CCA (K=2) or joint diagonalization (K>2)
   'maxIter',512*2, ... % max number of iterations
   'WDiffStop',1e-6, ... % stopping criterion
   'alpha0',0.1, ... % initial step size scaling (will be doubled for complex-valued)
   'mu',0.5, ... % initial step parameter for constrained algorithm update (default 0.5)
   'rho', 0.7, ... % correlation threshold for constrained algorithm (default 0.7)
   'gam',3 ... % lagrange multiplier step parameter (default 3)
   );

% load in user supplied options
options=getopt(options,varargin{:});

supplyA=~isempty(options.A); % set to true if user has supplied true mixing matrices
blowup = 1e3;
alphaScale=0.9; % alpha0 to alpha0*alphaScale when cost does not decrease
alphaMin=options.WDiffStop; % alpha0 will max(alphaMin,alphaScale*alpha0)
outputISI=false;

%% Create cell versions
% Faster than using the more natural 3-dimensional version
if iscell(options.A)
   K=length(options.A);
   A=zeros([size(options.A{1}) K]);
   for k=1:K
      A(:,:,k)=options.A{k};
   end
   options.A=A;

   clear A
end

if ~iscell(X)
   [N,T,K] = size(X); % the input format insures all datasets have equal number of samples
   if options.whiten
      [X,V]=whiten(X);
   end
   V_1 = zeros(N,N,K);
   for k=1:K
      V_1(:,:,k) = inv(V(:,:,k));
   end
      
   
   Rx=cell(K,K);
   for k1=1:K
      Rx{k1,k1}=1/T*(X(:,:,k1)*X(:,:,k1)');
      for k2=(k1+1):K
         Rx{k1,k2}=1/T*(X(:,:,k1)*X(:,:,k2)');
         Rx{k2,k1}=Rx{k1,k2}';
      end
   end
else % iscell
   [rowX,colX]=size(X);
   
   % then this is cell array with each entry being X{k} being a dataset
   % for now assume N is fixed
   X=X(:);
   
   K=max(colX,rowX);
   [N,T]=size(X{1});
  for k=1:K
     if T~=size(X{k},2)
        error('Each dataset must have the same number of samples.')
     end
  end % k
  
   if options.whiten
      [X,V_cell]=whiten(X);
      V=zeros(N,N,K);
      V_1 = zeros(N,N,K);
      for k=1:K
         V(:,:,k)=V_cell{k};
         V_1(:,:,k) = pinv(V(:,:,k));
      end
      clear V_cell
   end
   Rx=cell(K,K);
   for k1=1:K
      Rx{k1,k1}=1/T*(X{k1}*X{k1}');
      for k2=(k1+1):K
         Rx{k1,k2}=1/T*(X{k1}*X{k2}');
         Rx{k2,k1}=Rx{k1,k2}';
      end
   end
end

%% Initialize W
if ~isempty(options.W_init)
   W=options.W_init;
   
   if size(W,3)==1 && size(W,1)==N && size(W,2)==N
      W=repmat(W,[1,1,K]);
   end
   if options.whiten
      for k=1:K
         W(:,:,k)=W(:,:,k)/V(:,:,k);
      end
   end
else
    W=randn(N,N,K);
    for k = 1:K
       W(:,:,k)=vecnorm(W(:,:,k)')';
    end
end

%% Constraint flag == 1 ......... suchita
%% Check if parameters required for constrained IVA provided by user
if isempty(constraint_type)
    error('Provide type of constraint, 0 for mixing vector, 1 for source')
end
if isempty(guess_mat) % check for reference r_n
   error('Provide reference for constraining demixing vector')
end

num_guess=size(guess_mat,2); % number of references i.e. number of columns to be constrained
mu1 = (1-options.rho)*options.gam.* ones(num_guess,K); %lagrange multiplier (inequality constrain,mse criteria)

%%
num_W=size(W,1);
VV = zeros(N,N,K);
for k = 1 : K
    VV(:,:,k)=transpose(V_1(:,:,k));

    %Re-sort existing W based on correlation with matrix W:
    corr_w_guess = zeros(num_guess,num_W);
    for kl=1:num_guess
        r_n=guess_mat(:,kl);
        for lp=1:num_W
            w=W(lp,:,k).';
            if constraint_type == 0
                R_par = corrcoef(V_1(:,:,k)*w,r_n);
            else
                R_par = corrcoef(X(:,:,k)'*w,r_n);
            end
            corr_w_guess(kl,lp)=R_par(1,2);
        end
    end

    %We may need to use auction to chose order:
    [~,max_idx]=max(abs(corr_w_guess),[],2); % ; Added by Zois 

    if length(unique(max_idx)) ~= num_guess
        [colsol, ~] = auction((1-abs(corr_w_guess))');
        max_idx=colsol';
    end

    c = setxor((1:num_W).', max_idx);
    sort_order=[max_idx ;c];

    W(:,:,k)=W(sort_order,:,k);
end

%% When mixing matrix A supplied
% verbose is set to true
% outputISI can be computed if requested
% A matrix is conditioned by V if data is whitened
if supplyA
   % only reason to supply A matrices is to display running performance
   options.verbose=true;
   if nargout>3
      outputISI=true;
      isi=nan(1,options.maxIter);
   end
   if options.whiten
      for k = 1:K
         options.A(:,:,k) = V(:,:,k)*options.A(:,:,k);
      end
   end
end

%% Check rank of data-covariance matrix
% should be full rank, if not we inflate
% this is ad hoc and no robustness is assured by diagonal loading procedure
Rxall=cell2mat(Rx);
r=rank(Rxall);
if r<(N*K)
   % inflate Rx
   eigRx=svd(Rxall);
   Rxall=Rxall+eigRx(r)*eye(N*K); %diag(linspace(0.1,0.9,N*K)); % diagonal loading
   Rx=mat2cell(Rxall,N*ones(K,1),N*ones(K,1));
end

%% Initialize some local variables
cost=nan(1,options.maxIter);
cost_const=K*log(2*pi*exp(1)); % local constant
condition_number = nan(N,1);

%% Initializations based on real-valued
grad=zeros(N,K);
% if opt_approach==2
%   H=zeros(N*K,N*K);
% end

%% Main Iteration Loop
for iter = 1:options.maxIter
   termCriterion=0;
   
   % Some additional computations of performance via ISI when true A is supplied
   if supplyA
      [amari_avg_isi,amari_joint_isi,~]=bss_isi(W,options.A);
      if outputISI
         isi(iter)=amari_joint_isi;
      end
   end
   
   W_old=W; % save current W as W_old
   cost(iter)=0;
   for k=1:K
      cost(iter)=cost(iter)-log(abs(det(W(:,:,k))));
   end
   
   Q=0; R=0;
   %% Loop over each SCV
   for n=1:N
%       Wn=W(n,:,:);
%       Wn=conj(Wn(:));
            
      % Efficient version of Sigma_n=Yn*Yn'/T;
      Sigma_n=eye(K); %
      for k1=1:K
         for k2=k1:K
            Sigma_n(k1,k2)=W(n,:,k1)*Rx{k1,k2}*W(n,:,k2)';
            if k1~=k2
               Sigma_n(k2,k1)=Sigma_n(k1,k2)';
            end
         end % k2
      end %k1
      
      cost(iter)=cost(iter)+0.5*(cost_const+log(det(Sigma_n)));
      inv_Sigma_n=inv(Sigma_n);
      
      [hnk,Q,R]=decouple_trick(W,n,Q,R);
      
      for k=1:K         
         % Analytic derivative of cost function with respect to vn
         % Code below is efficient implementation of computing the gradient, that is
         % independent of T
         grad(:,k)=-hnk(:,k)/(W(n,:,k)*hnk(:,k));
         
        for kk=1:K
           grad(:,k)=grad(:,k)+ ...
              Rx{k,kk}*W(n,:,kk)'*inv_Sigma_n(kk,k);
        end
        
        wnk=W(n,:,k)';
        if options.opt_approach==1
           gradNorm=vecnorm(grad(:,k)); % normalize gradient
           gradNormProj=vecnorm(gradNorm-wnk'*gradNorm*wnk); % non-colinear direction normalized
             %% Add constraint...... suchita
            % constraint_type, 0 for mixing vector 1 for source 
                
           num_guess= size(guess_mat, 2);
            
            if (n <= num_guess)
                r_n=guess_mat(:,n);
                if constraint_type == 0 % constrain de-mixing vector, w_n
                    %Modify r_i:
                    r_n_mu=VV(:,:,k)*r_n;
                    r_n_mu=r_n_mu/norm(r_n_mu);
                else % constrain source y_n
                    r_n_mu = r_n;
                    r_n_mu=r_n_mu/norm(r_n_mu);
                end


                if constraint_type == 0 % constrain mixing vector
                    R_par = corrcoef(V_1(:,:,k)*wnk,r_n); %correlation
                else % constrain Y (source)
                    R_par = corrcoef((X(:,:,k)'*wnk),r_n);
                end
                r_corr=abs(R_par(1,2));
                mu1_new(n,k)=sign(R_par(1,2))*(max(0,mu1(n,k) + options.gam*(options.rho-r_corr)));
                mu1_old(n,k) = mu1(n,k);
                mu1(n,k) = mu1_new(n,k);
                
                if constraint_type == 0
                    dW_C = (mu1(n,k).*r_n_mu); % derivative of constrain term
                else
                    dW_C = mu1(n,k).*X(:,:,k)*r_n_mu;
                end
                W(n,:,k)= (wnk - options.alpha0.*(gradNormProj - dW_C));
            else
                W(n,:,k)=(wnk-options.alpha0*gradNormProj)';
            end                       

           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %%%%% The following computations are potentially unnecessary
           %%%%% see that the quasi-newton approach does not do this
           %%%%% step.
           for kk=1:K
              Sigma_n(k,kk)=W(n,:,k)*Rx{k,kk}*W(n,:,kk)'; % = Yn*Yn'/T;
           end
           Sigma_n(:,k)=Sigma_n(k,:)';

           inv_Sigma_n=inv(Sigma_n);

           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        end % if opt_approach == 1
      end % k
   end %n
   for k=1:K
        W(:,:,k)=vecnorm(W(:,:,k)')';
   end
   
   cost(iter) = cost(iter) - (sum(sum(mu1_new.^2 - mu1_old.^2)))/(2*options.gam);
  
%    cost(iter)=cost(iter)/(N*K);
   for k=1:K
      termCriterion = max(termCriterion,max(1-abs(diag(W_old(:,:,k)*W(:,:,k)'))));
   end % k
   
   %% Decrease step size alpha if cost increased from last iteration
   if iter>1
      if cost(iter)>cost(iter-1)
         options.alpha0=max(alphaMin,alphaScale*options.alpha0);               
      end
   end
   
   %% Check the termination condition
   if termCriterion < options.WDiffStop || iter == options.maxIter || options.alpha0 == alphaMin
      break;
   elseif termCriterion > blowup || isnan(cost(iter))
      for k = 1:K
         W(:,:,k) = eye(N) + 0.1*randn(N);
      end
      if options.verbose
         fprintf('\n W blowup, restart with new initial value.');
      end
   end
   
   %% Display Iteration Information
   if options.verbose
      if supplyA
         fprintf('\n Step %d: W change: %f, Cost: %f, Avg ISI: %f, Joint ISI: %f',  ...
            iter, termCriterion,cost(iter),amari_avg_isi,amari_joint_isi);
      else
         fprintf('\n Step %d: W change: %f, Cost: %f', iter, termCriterion,cost(iter));
      end
   end % options.verbose
end % iter

%% Finish Display
if iter==1 && options.verbose
   if supplyA
      fprintf('\n Step %d: W change: %f, Cost: %f, Avg ISI: %f, Joint ISI: %f',  ...
         iter, termCriterion,cost(iter),amari_avg_isi,amari_joint_isi);
   else
      fprintf('\n Step %d: W change: %f, Cost: %f', iter, termCriterion,cost(iter));
   end
end % options.verbose

if options.verbose
   fprintf('\n');
end

%% Clean-up Outputs
cost=cost(1:iter);
if outputISI
   isi=isi(1:iter);
end

if options.whiten
   for k=1:K
      W(:,:,k)=W(:,:,k)*V(:,:,k);
   end
else % no prewhitening
   %% Scale demixing vectors to generate unit variance sources
   for n=1:N
      for k=1:K
         W(n,:,k)=W(n,:,k)/sqrt(W(n,:,k)*Rx{k,k}*W(n,:,k)');
      end
   end
end

%% Resort order of SCVs
% Order the components from most to least ill-conditioned

% First, compute the data covariance matrices (by undoing any whitening)
if options.whiten
   for k1=1:K
      Rx{k1,k1}=(V(:,:,k1)\Rx{k1,k1})/V(:,:,k1)';
      for k2=(k1+1):K
         Rx{k1,k2}=(V(:,:,k1)\Rx{k1,k2})/V(:,:,k2)';
         Rx{k2,k1}=Rx{k1,k2}';
      end % k2
   end % k1
end

% Second, compute the determinant of the SCVs
detSCV=zeros(1,N);
Sigma_N=zeros(K,K,N);
for n=1:N
   % Efficient version of Sigma_n=Yn*Yn'/T;
   Sigma_n=zeros(K); %
   for k1=1:K
      for k2=k1:K
         Sigma_n(k1,k2)=W(n,:,k1)*Rx{k1,k2}*W(n,:,k2)';
         Sigma_n(k2,k1)=Sigma_n(k1,k2)';
      end % k2
   end %k3
   Sigma_N(:,:,n)=Sigma_n;
   condition_number(n,1) = cond(Sigma_N(:,:,n));
   detSCV(n)=det(Sigma_n);
end

% Third, sort and apply
[~,isort]=sort(detSCV);
Sigma_N=Sigma_N(:,:,isort);
for k=1:K
   W(:,:,k)=W(isort,:,k);
end

%% Cast output
if iscell(X)
   Wmat=W;
   W=cell(K,1);
   for k=1:K
      W{k}=Wmat(:,:,k);
   end
end

%% Return
end
% END OF IVA_MG_OPTS

function test_iva_g_constrained
%% built-in test function for iva_second_order
N = 10;  % number of sources
K = 10;  % number of groups
T = 1000;   % sample size

% generate the mixtures
S = zeros(N,T,K);
for n = 1 : N
    Q = rand(K,K);
    Sigma = Q*Q';
    mu = zeros(1,K);
    s = mvnrnd(mu,Sigma,T)';
    for k = 1 : K
        S(n,:,k) = s(k,:);
        S(n,:,k) = S(n,:,k) - mean(S(n,:,k));
        S(n,:,k) = S(n,:,k)/sqrt(var(S(n,:,k)));
    end
end
A = randn(N,N,K);
x = zeros(N,T,K);
step = [-1*ones(1,ceil(N/2)),ones(1,N-ceil(N/2))];
for k=1:K
    A(:,1,k) = step + 0.1*randn(1,N); % add a known reference signal to one column
    x(:,:,k) = A(:,:,k)*S(:,:,k);
end

%% Example call to iva_l_sos_constrained to constrain columns of mixing matrix
W=iva_g_constrained(x,step',0,'A',A);
% show results
figure;
T1 = zeros(N,N);
for k = 1 : K
    Tk = W(:,:,k)*A(:,:,k);
    Tk = abs(Tk);
    for n = 1 : N
        Tk(n,:) = Tk(n,:)/max(abs(Tk(n,:)));
    end
    T1 = T1 + Tk/K;
end
P=zeros(N); 
[~,imax]=max(T1);
ind=sub2ind([N N],1:N,imax);
P(ind)=1;
T1=P*T1;
imagesc(1:N,1:N,T1)
caxis([0 1])
axis xy
colormap('bone')
title('joint global matrix')
colorbar
disp('Ideally image is identity matrix.')

%% Example call to iva_l_sos_constrained to constrain sources
W=iva_g_constrained(x,mean(S(1:3,:,:),3)',1,'A',A); 
figure;
% show results
T1 = zeros(N,N);
for k = 1 : K
    Tk = W(:,:,k)*A(:,:,k);
    Tk = abs(Tk);
    for n = 1 : N
        Tk(n,:) = Tk(n,:)/max(abs(Tk(n,:)));
    end
    T1 = T1 + Tk/K;
end
P=zeros(N); 
[~,imax]=max(T1);
ind=sub2ind([N N],1:N,imax);
P(ind)=1;
T1=P*T1;
imagesc(1:N,1:N,T1)
caxis([0 1])
axis xy
colormap('bone')
title('joint global matrix')
colorbar
disp('Ideally image is identity matrix.')
end

%%
function properties = getopt(properties,varargin)
    %GETOPT - Process paired optional arguments as 'prop1',val1,'prop2',val2,...
    %
    %   getopt(properties,varargin) returns a modified properties structure,
    %   given an initial properties structure, and a list of paired arguments.
    %   Each argumnet pair should be of the form property_name,val where
    %   property_name is the name of one of the field in properties, and val is
    %   the value to be assigned to that structure field.
    %
    %   No validation of the values is performed.
    %%
    % EXAMPLE:
    %   properties = struct('zoom',1.0,'aspect',1.0,'gamma',1.0,'file',[],'bg',[]);
    %   properties = getopt(properties,'aspect',0.76,'file','mydata.dat')
    % would return:
    %   properties =
    %         zoom: 1
    %       aspect: 0.7600
    %        gamma: 1
    %         file: 'mydata.dat'
    %           bg: []
    %
    % Typical usage in a function:
    %   properties = getopt(properties,varargin{:})

    % Function from
    % http://mathforum.org/epigone/comp.soft-sys.matlab/sloasmirsmon/bp0ndp$crq5@cui1.lmms.lmco.com

    % dgleich
    % 2003-11-19
    % Added ability to pass a cell array of properties

    if ~isempty(varargin) && (iscell(varargin{1}))
       varargin = varargin{1};
    end;

    % Process the properties (optional input arguments)
    prop_names = fieldnames(properties);
    TargetField = [];
    for ii=1:length(varargin)
       arg = varargin{ii};
       if isempty(TargetField)
          if ~ischar(arg)
             error('Property names must be character strings');
          end
          %f = find(strcmp(prop_names, arg));
          if isempty(find(strcmp(prop_names, arg),1)) %length(f) == 0
             error('%s ',['invalid property ''',arg,'''; must be one of:'],prop_names{:});
          end
          TargetField = arg;
       else
          properties.(TargetField) = arg;
          TargetField = '';
       end
    end
    if ~isempty(TargetField)
       error('Property names and values must be specified in pairs.');
    end
end

%%
function mag=vecmag(vec,varargin)
% mag=vecmag(vec)
% or
% mag=vecmag(v1,v2,...,vN)
%
% Computes the vector 2-norm or magnitude of vector. vec has size n by m
% represents m vectors of length n (i.e. m column-vectors). Routine avoids
% potential mis-use of norm built-in function. Routine is faster than
% calling sqrt(dot(vec,vec)) -- but equivalent.
    if nargin==1
       mag=sqrt(sum(vec.*conj(vec)));
    else
       mag=vec.*conj(vec);
       for ii=1:length(varargin)
          mag=mag+varargin{ii}.*conj(varargin{ii});
       end
       mag=sqrt(mag);
    end
end

%%
function [uvec,mag]=vecnorm(vec)
% [vec,mag]=vecnorm(vec)
% Returns the vector normalized by 2-norm or magnitude of vector.
% vec has size n by m represents m vectors of length n (i.e. m
% column-vectors).
    [n,m]=size(vec);
    if n==1
       disp('vecnorm operates on column vectors, input appears to have dimension of 1')
    end

    uvec=zeros(n,m);
    mag=vecmag(vec); % returns a 1 x m row vector
    for ii=1:size(vec,1)
       uvec(ii,:)=vec(ii,:)./mag;
    end
    % Equivalent to: uvec=vec./repmat(mag,size(vec,1),1);

    % Which implementation is optimal depends on optimality criterion (memory
    % vs. speed), this version uses the former criterion.
end

%% Decoupling trick
% h=decouple_trick(W,n)
% h=decouple_trick(W,n,1)
% [h,invQ]=decouple_trick(W,n,invQ)
% [h,Q,R]=decouple_trick(W,n,Q,R)
%
% Computes the h vector for the decoupling trick [1] of the nth row of W. W
% can be K 'stacked' square matrices, i.e., W has dimensions N x N x K.
% The output vector h will be formatted as an N x K matrix.  There are many
% possible methods for computing h.  This routine provides four different
% (but of course related) methods depending on the arguments used.
%
% Method 1:
% h=decouple_trick(W,n)
% h=decouple_trick(W,n,0)
% -Both calls above will result in the same algorithm, namely the QR
% algorithm is used to compute h.
%
% Method 2:
% h=decouple_trick(W,n,~), where ~ is anything
% -Calls the projection method.
%
% Method 3:
% [h,invQ]=decouple_trick(W,n,invQ)
% -If two output arguments are specified then the recursive algorithm
% described in [2].  It is assumed that the decoupling will be performed in
% sequence see the demo subfunction for details.
% An example call sequence:
%  [h1,invQ]=decouple_trick(W,1);
%  [h2,invQ]=decouple_trick(W,2,invQ);
% 
% Method 4:
% [h,Q,R]=decouple_trick(W,n,Q,R)
% -If three output arguments are specified then a recursive QR algorithm is
% used to compute h.
% An example call sequence:
%  [h1,Q,R]=decouple_trick(W,1);
%  [h2,Q,R]=decouple_trick(W,2,Q,R);
%
% See the subfunction demo_decoupling_trick for more examples.  The demo
% can be executed by calling decouple_trick with no arguments, provides a
% way to compare the speed and determine the accuracy of all four
% approaches.  
%
% Note that methods 2 & 3 do not normalize h to be a unit vector.  For
% optimization this is usually not of interest.  If it is then set the
% variable boolNormalize to true.
%
% Main References:
% [1] X.-L. Li & X.-D. Zhang, "Nonorthogonal Joint Diagonalization Free of Degenerate Solution," IEEE Trans. Signal Process., 2007, 55, 1803-1814
% [2] X.-L. Li & T. Adali, "Independent component analysis by entropy bound minimization," IEEE Trans. Signal Process., 2010, 58, 5151-5164
%
% Coded by Matthew Anderson (matt dot anderson at umbc dot edu)

% Version 01 - 20120919 - Initial publication

function [h,invQ,R]=decouple_trick(W,n,invQ,R)
    if nargin==0
       help decouple_trick
       demo_decouple_trick
       return
    end
    if nargin==1
       help decouple_trick
       error('Not enough inputs -- see displayed help above.')
    end
    [M,N,K]=size(W);
    if M~=N
       error('Assuming W is square matrix.')
    end
    h=zeros(N,K);

    % enables an additional computation that is usually not necessary if the
    % derivative is of  interest, it is only necessary so that sqrt(det(W*W'))
    % = sqrt(det(Wtilde*Wtilde'))*abs(w'*h) holds.  Furthermore, it is only
    % necessary when using the recursive or projection methods.
    %
    % a user might wish to enable the calculation by setting the quantity below
    % to true 
    boolNormalize=false; 

    if nargout==3
       % use QR recursive method
       % [h,Qnew,Rnew]=decouple_trick(W,n,Qold,Rold)
       if n==1
          invQ=zeros(N,N,K);
          R=zeros(N,N-1,K);
       end
       for k=1:K
          if n==1
             Wtilde=W(2:N,:,k);
             [invQ(:,:,k),R(:,:,k)]=qr(Wtilde');
          else
             n_last=n-1;
             e_last = zeros(N-1,1);
             e_last(n_last) = 1;         
             [invQ(:,:,k),R(:,:,k)]=qrupdate(invQ(:,:,k),R(:,:,k),-W(n,:,k)',e_last);
             [invQ(:,:,k),R(:,:,k)]=qrupdate(invQ(:,:,k),R(:,:,k),W(n_last,:,k)',e_last);
          end
          h(:,k)=invQ(:,end,k); % h should be orthogonal to W(nout,:,k)'
       end   
    elseif nargout==2
       % use recursive method
       % [h,invQ]=decouple_trick(W,n,invQ), for any value of n=1, ..., N
       % [h,invQ]=decouple_trick(W,1), when n=1

       if n==1
          invQ=zeros(N-1,N-1,K);
       end
       % Implement a faster approach to calculating h.
       for k=1:K
          if n==1
             Wtilde=W(2:N,:,k);
             invQ(:,:,k)=inv(Wtilde*Wtilde');
          else
             if nargin<3
                help decouple_trick
                error('Need to supply invQ for recursive approach.')
             end
             [Mq,Nq,Kq]=size(invQ);
             if Mq~=(N-1) || Nq~=(N-1) || Kq~=K
                help decouple_trick
                error('Input invQ does not have the expected dimensions.')
             end
             n_last=n-1;
             Wtilde_last=W([(1:n_last-1) (n_last+1:N)],:,k);         
             w_last=W(n_last,:,k)';
             w_current=W(n,:,k)';
             c = Wtilde_last*(w_last - w_current);
             c(n_last) = 0.5*( w_last'*w_last - w_current'*w_current );
             %e_last = zeros(N-1,1);
             %e_last(n_last) = 1;
             temp1 = invQ(:,:,k)*c;
             temp2 = invQ(:,n_last,k);
             inv_Q_plus = invQ(:,:,k) - temp1*temp2'/(1+temp1(n_last));

             temp1 = inv_Q_plus'*c;
             temp2 = inv_Q_plus(:,n_last);
             invQ(:,:,k) = inv_Q_plus - temp2*temp1'/(1+c'*temp2);
             % inv_Q is Hermitian
             invQ(:,:,k) = (invQ(:,:,k)+invQ(:,:,k)')/2;
          end

          temp1 = randn(N, 1);
          Wtilde = W([(1:n-1) (n+1:N)],:,k);
          h(:,k) = temp1 - Wtilde'*invQ(:,:,k)*Wtilde*temp1;
       end
       if boolNormalize
          h=vecnorm(h);
       end
    elseif nargin==2 || invQ==0    
       % use (default) QR approach
       % h=decouple_trick(W,n)
       % h=decouple_trick(W,n,0)
       for k=1:K
          [Q,~]=qr(W([(1:n-1) (n+1:N)],:,k)');
          h(:,k)=Q(:,end); % h should be orthogonal to W(nout,:,k)'
       end
    else % use projection method
       % h=decouple_trick(W,n,~), ~ is anything
       for k=1:K
          temp1 = randn(N, 1);
          Wtilde = W([(1:n-1) (n+1:N)],:,k);
          h(:,k) = temp1 - Wtilde'*((Wtilde*Wtilde')\Wtilde)*temp1;
       end
       if boolNormalize
          h=vecnorm(h);
       end
    end
end

%% Function for ISI
function [isi,isiGrp,success,G]=bss_isi(W,A,s,Nuse)
% Non-cell inputs:
% isi=bss_isi(W,A) - user provides W & A where x=A*s, y=W*x=W*A*s
% isi=bss_isi(W,A,s) - user provides W, A, & s
%
% Cell array of matrices:
% [isi,isiGrp]=bss_isi(W,A) - W & A are cell array of matrices
% [isi,isiGrp]=bss_isi(W,A,s) - W, A, & s are cell arrays
%
% 3-d Matrices:
% [isi,isiGrp]=bss_isi(W,A) - W is NxMxK and A is MxNxK
% [isi,isiGrp]=bss_isi(W,A,s) - S is NxTxK (N=#sources, M=#sensors, K=#datasets)
%
% Measure of quality of separation for blind source separation algorithms.
% W is the estimated demixing matrix and A is the true mixing matrix.  It should be noted
% that rows of the mixing matrix should be scaled by the necessary constants to have each
% source have unity variance and accordingly each row of the demixing matrix should be
% scaled such that each estimated source has unity variance.
%
% ISI is the performance index given in Complex-valued ICA using second order statisitcs
% Proceedings of the 2004 14th IEEE Signal Processing Society Workshop, 2004, 183-192
%
% Normalized performance index (Amari Index) is given in Choi, S.; Cichocki, A.; Zhang, L.
% & Amari, S. Approximate maximum likelihood source separation using the natural gradient
% Wireless Communications, 2001. (SPAWC '01). 2001 IEEE Third Workshop on Signal
% Processing Advances in, 2001, 235-238.
%
% Note that A is p x M, where p is the number of sensors and M is the number of signals
% and W is N x p, where N is the number of estimated signals.  Ideally M=N but this is not
% guaranteed.  So if N > M, the algorithm has estimated more sources than it "should", and
% if M < N the algorithm has not found all of the sources.  This meaning of this metric is
% not well defined when averaging over cases where N is changing from trial to trial or
% algorithm to algorithm.

% Some examples to consider
% isi=bss_isi(eye(n),eye(n))=0
%
% isi=bss_isi([1 0 0; 0 1 0],eye(3))=NaN
%


% Should ideally be a permutation matrix with only one non-zero entry in any row or
% column so that isi=0 is optimal.

% generalized permutation invariant flag (default=false), only used when nargin<3
    gen_perm_inv_flag=false;
    success=true;

    Wcell=iscell(W);
    if nargin<2
       Acell=false;
    else
       Acell=iscell(A);
    end
    if ~Wcell && ~Acell
       if ndims(W)==2 && ndims(A)==2
          if nargin==2
             % isi=bss_isi(W,A) - user provides W & A

             % Traditional Metric, user provided W & A separately
             G=W*A;
             [N,M]=size(G);
             Gabs=abs(G);
             if gen_perm_inv_flag
                % normalization by row
                max_G=max(Gabs,[],2);
                Gabs=repmat(1./max_G,1,size(G,2)).*Gabs;
             end
          elseif nargin==3
             % Equalize energy associated with each estimated source and true
             % source.
             %
             % y=W*A*s; 
             % snorm=D*s; where snorm has unit variance: D=diag(1./std(s,0,2))
             % Thus: y=W*A*inv(D)*snorm
             % ynorm=U*y; where ynorm has unit variance: U=diag(1./std(y,0,2))
             % Thus: ynorm=U*W*A*inv(D)*snorm=G*snorm and G=U*W*A*inv(D)

             y=W*A*s;
             D=diag(1./std(s,0,2));         
             U=diag(1./std(y,0,2));
             G=U*W*A/D; % A*inv(D)
             [N,M]=size(G);
             Gabs=abs(G);
          else
             error('Not acceptable.')
          end

          isi=0;
          for n=1:N
             isi=isi+sum(Gabs(n,:))/max(Gabs(n,:))-1;
          end
          for m=1:M
             isi=isi+sum(Gabs(:,m))/max(Gabs(:,m))-1;
          end
          isi=isi/(2*N*(N-1));
          isiGrp=NaN;
          success=NaN;
       elseif ndims(W)==3 && ndims(A)==3
          % IVA/GroupICA/MCCA Metrics
          % For this we want to average over the K groups as well as provide the additional
          % measure of solution to local permutation ambiguity (achieved by averaging the K
          % demixing-mixing matrices and then computing the ISI of this matrix).
          [N,M,K]=size(W);      
          if M~=N
             error('This more general case has not been considered here.')
          end
          L=M;

          isi=0;
          GabsTotal=zeros(N,M);
          G=zeros(N,M,K);
          for k=1:K
             if nargin<=2
                % Traditional Metric, user provided W & A separately
                Gk=W(:,:,k)*A(:,:,k);
                Gabs=abs(Gk);
                if gen_perm_inv_flag
                   % normalization by row
                   max_G=max(Gabs,[],2);
                   Gabs=repmat(1./max_G,1,size(Gabs,2)).*Gabs;
                end
             else %if nargin==3
                % Equalize energy associated with each estimated source and true
                % source.
                %
                % y=W*A*s;
                % snorm=D*s; where snorm has unit variance: D=diag(1./std(s,0,2))
                % Thus: y=W*A*inv(D)*snorm
                % ynorm=U*y; where ynorm has unit variance: U=diag(1./std(y,0,2))
                % Thus: ynorm=U*W*A*inv(D)*snorm=G*snorm and G=U*W*A*inv(D)
                yk=W(:,:,k)*A(:,:,k)*s(:,:,k);
                Dk=diag(1./std(s(:,:,k),0,2));
                Uk=diag(1./std(yk,0,2));
                Gk=Uk*W(:,:,k)*A(:,:,k)/Dk;

                Gabs=abs(Gk);         
             end
             G(:,:,k)=Gk;

             if nargin>=4
                Np=Nuse;
                Mp=Nuse;
                Lp=Nuse;
             else
                Np=N;
                Mp=M;
                Lp=L;
             end

             % determine if G is success by making sure that the location of maximum magnitude in
             % each row is unique.
             if k==1
                [~,colMaxG]=max(Gabs,[],2);
                if length(unique(colMaxG))~=Np
                   % solution is failure in strictest sense
                   success=false;
                end
             else
                [~,colMaxG_k]=max(Gabs,[],2);
                if ~all(colMaxG_k==colMaxG)
                   % solution is failure in strictest sense
                   success=false;
                end
             end

             GabsTotal=GabsTotal+Gabs;

             for n=1:Np
                isi=isi+sum(Gabs(n,:))/max(Gabs(n,:))-1;
             end
             for m=1:Mp
                isi=isi+sum(Gabs(:,m))/max(Gabs(:,m))-1;
             end
          end
          isi=isi/(2*Np*(Np-1)*K);

          Gabs=GabsTotal;
          if gen_perm_inv_flag
             % normalization by row
             max_G=max(Gabs,[],2);
             Gabs=repmat(1./max_G,1,size(Gabs,2)).*Gabs;
          end
    %       figure; imagesc(Gabs); colormap('bone'); colorbar
          isiGrp=0;
          for n=1:Np
             isiGrp=isiGrp+sum(Gabs(n,:))/max(Gabs(n,:))-1;
          end
          for m=1:Mp
             isiGrp=isiGrp+sum(Gabs(:,m))/max(Gabs(:,m))-1;
          end
          isiGrp=isiGrp/(2*Lp*(Lp-1));
       else
          error('Need inputs to all be of either dimension 2 or 3')
       end
    elseif Wcell && Acell
       % IVA/GroupICA/MCCA Metrics
       % For this we want to average over the K groups as well as provide the additional
       % measure of solution to local permutation ambiguity (achieved by averaging the K
       % demixing-mixing matrices and then computing the ISI of this matrix).

       K=length(W);
       N=0; M=0;
       Nlist=zeros(K,1);
       for k=1:K
          Nlist(k)=size(W{k},1);
          N=max(size(W{k},1),N);
          M=max(size(A{k},2),M);
       end
       commonSources=false; % limits the ISI to first min(Nlist) sources
       if M~=N
          error('This more general case has not been considered here.')
       end
       L=M;

       % To make life easier below lets sort the datasets to have largest
       % dataset be in k=1 and smallest at k=K;
       [Nlist,isort]=sort(Nlist,'descend');
       W=W(isort);
       A=A(isort);
       if nargin > 2
          s=s(isort);
       end
       G=cell(K,1);
       isi=0;
       if commonSources
          minN=min(Nlist);
          GabsTotal=zeros(minN);
          Gcount=zeros(minN);
       else
          GabsTotal=zeros(N,M);
          Gcount=zeros(N,M);
       end
       for k=1:K
          if nargin==2
             % Traditional Metric, user provided W & A separately
             G{k}=W{k}*A{k};
             Gabs=abs(G{k});
             if gen_perm_inv_flag
                % normalization by row
                max_G=max(Gabs,[],2);
                Gabs=repmat(1./max_G,1,size(Gabs,2)).*Gabs;
             end
          elseif nargin>=3
             % Equalize energy associated with each estimated source and true
             % source.
             %
             % y=W*A*s;
             % snorm=D*s; where snorm has unit variance: D=diag(1./std(s,0,2))
             % Thus: y=W*A*inv(D)*snorm
             % ynorm=U*y; where ynorm has unit variance: U=diag(1./std(y,0,2))
             % Thus: ynorm=U*W*A*inv(D)*snorm=G*snorm and G=U*W*A*inv(D)
             yk=W{k}*A{k}*s{k};
             Dk=diag(1./std(s{k},0,2));
             Uk=diag(1./std(yk,0,2));
             G{k}=Uk*W{k}*A{k}/Dk;

             Gabs=abs(G{k});
          else
             error('Not acceptable.')
          end

          if commonSources
             Nk=minN;
             Gabs=Gabs(1:Nk,1:Nk);
          elseif nargin>=4
             commonSources=true;
             Nk=Nuse;
             minN=Nk;
          else
             Nk=Nlist(k);
          end      

          if k==1
             [~,colMaxG]=max(Gabs(1:Nk,1:Nk),[],2);
             if length(unique(colMaxG))~=Nk
                % solution is a failure in a strict sense
                success=false;
             end
          elseif success
             if nargin>=4
                [~,colMaxG_k]=max(Gabs(1:Nk,1:Nk),[],2);
             else
                [~,colMaxG_k]=max(Gabs,[],2);
             end
             if ~all(colMaxG_k==colMaxG(1:Nk))
                % solution is a failure in a strict sense
                success=false;
             end
          end

          if nargin>=4
             GabsTotal(1:Nk,1:Nk)=GabsTotal(1:Nk,1:Nk)+Gabs(1:Nk,1:Nk);
          else
             GabsTotal(1:Nk,1:Nk)=GabsTotal(1:Nk,1:Nk)+Gabs;
          end
          Gcount(1:Nk,1:Nk)=Gcount(1:Nk,1:Nk)+1;
          for n=1:Nk
             isi=isi+sum(Gabs(n,:))/max(Gabs(n,:))-1;
          end
          for m=1:Nk
             isi=isi+sum(Gabs(:,m))/max(Gabs(:,m))-1;
          end
          isi=isi/(2*Nk*(Nk-1));     
       end  

       if commonSources
          Gabs=GabsTotal;
       else
          Gabs=GabsTotal./Gcount;
       end
       % normalize entries into Gabs by the number of datasets
       % contribute to each entry

       if gen_perm_inv_flag
          % normalization by row
          max_G=max(Gabs,[],2);
          Gabs=repmat(1./max_G,1,size(Gabs,2)).*Gabs;
       end
       isiGrp=0;

       if commonSources
          for n=1:minN
             isiGrp=isiGrp+sum(Gabs(n,1:minN))/max(Gabs(n,1:minN))-1;
          end
          for m=1:minN
             isiGrp=isiGrp+sum(Gabs(1:minN,m))/max(Gabs(1:minN,m))-1;
          end
          isiGrp=isiGrp/(2*minN*(minN-1));
       else
          for n=1:Nk
             isiGrp=isiGrp+sum(Gabs(n,:))/max(Gabs(n,:))-1;
          end
          for m=1:Nk
             isiGrp=isiGrp+sum(Gabs(:,m))/max(Gabs(:,m))-1;
          end
          isiGrp=isiGrp/(2*L*(L-1));
       end

    else
       % Have not handled when W is cell and A is single matrix or vice-versa.  Former makes
       % sense when you want performance of multiple algorithms for one mixing matrix, while
       % purpose of latter is unclear.
    end

end

%%
function [colsol, rowsol] = auction(assignCost)

%function [colsol, rowsol] = auction(assignCost, guard)
%AUCTION : performs assignment using Bertsekas' auction algorithm
%          The auction is 1-sided and uses a fixed epsilon.
%  colsol = auction(assignCost) returns column assignments: colsol(j) gives the
%           row assigned to column j.
%
%           assignCost is an m x n matrix of costs for associating each row
%           with each column.  m >= n (rectangular assignment is allowed).
%           Auction finds the assignment that minimizes the costs.
%
%  colsol = auction(assignCost, guard) sets the cost of column
%           non-assignment = guard.  All assignments will have cost < guard.
%           A column will not be assigned if the savings from eliminating it
%           exceeds the guard cost.  colsol(j) = 0 indicates that the optimal
%           solution does not assign column j to any row.
%
%  [colsol,rowsol] = auction(assignCost) also returns the row assignments:
%           rowsol(i) gives the column assigned to row j.
%
%  Reference
%  Bertsekas, D. P., "The Auction Algorithm: A Distributed Relaxation Method
%  for the Assignment Problem," Annals of Operations Research, 14, 1988, pp.
%  103-123.

% Mark Levedahl
% 
%=================================================================================

[m,n] = size(assignCost);

if m < n
	error('cost matrix must have no more columns than rows.')
end

% augment cost matrix with a guard row if specified.
m0 = m;
% if isfinite(guard) %%% DONE BY JS
% 	m = m+1;
% 	assignCost(m,:) = guard;
% end

% init return arrays
colsol = zeros(1,n);
rowsol = zeros(m,1);
price = zeros(m,1);
EPS = sqrt(eps) / (n+1);

% 1st step is a full parallel solution.  Get bids for all columns
jp = 1:n;
f = assignCost;
[b1,ip] = min(f);                   % cost and row of best choice
f(ip + m*(0:n-1)) = inf;            % eliminate best from contention
bids = min(f) - b1;                 % cost of runner up choice hence bid

% Arrange bids so highest (best) are last and will overwrite the lower bids.
[tmp,ibid] = sort(bids(:));

% Now assign best bids (lesser bids are overwritten by better ones).
price(ip(ibid)) = price(ip(ibid)) + EPS + tmp;
rowsol(ip(ibid)) = jp(ibid);        % do row assignments
iy = find(rowsol);
colsol(rowsol(iy)) = iy;            % map to column assignments

% The guard row cannot be assigned (always available)
if m0 < m
	price(m) = 0;
	rowsol(m) = 0;
end

% Now Continue with non-parallel code handling any contentions.
while ~all(colsol)
	for jp = find(~colsol)
		f = assignCost(:,jp) + price;   % costs
		[b1,ip] = min(f);               % cost and row of best choice
		if ip > m0
			colsol(jp) = m;
		else
			f(ip) = inf;                    % eliminate from contention
			price(ip) = price(ip) + EPS + min(f) - b1; % runner up choice hence bid
			if rowsol(ip)                   % take the row away if already assigned
				colsol(rowsol(ip)) = 0;
			end
			rowsol(ip) = jp;                % update row and column assignments
			colsol(jp) = ip;
		end % if ip == m
	end
end

% screen out infeasible assignments
if m > m0
	colsol(colsol == m) = 0;
	rowsol(m) = [];
end
end
%% Function for whitening 
function [z,V,U]=whiten(x)
% [z,V,U]=whiten(x)
% 
% Whitens the data vector so that E{zz'}=I, where z=V*x.

    if ~iscell(x)
       [N,T,K]=size(x);
       if K==1
          % Step 1. Center the data.
          x=bsxfun(@minus,x,mean(x,2));

          % Step 2. Form MLE of data covariance.
          covar=x*x'/T;

          % Step 3. Eigen decomposition of covariance.
          [eigvec, eigval] = eig (covar);

          % Step 4. Forming whitening transformation.
          V=sqrt(eigval) \ eigvec';
          U=eigvec * sqrt(eigval);

          % Step 5. Form whitened data
          z=V*x;
       else
          K=size(x,3);
          z=zeros(N,T,K);
          V=zeros(N,N,K);
          U=zeros(N,N,K);
          for k=1:K
             % Step 1. Center the data.
             xk=bsxfun(@minus,x(:,:,k),mean(x(:,:,k),2));

             % Step 2. Form MLE of data covariance.
             covar=xk*xk'/T;

             % Step 3. Eigen decomposition of covariance.
             [eigvec, eigval] = eig (covar);

             % Step 4. Forming whitening transformation.
             V(:,:,k)=sqrt(eigval) \ eigvec';
             U(:,:,k)=eigvec * sqrt(eigval);

             % Step 5. Form whitened data
             z(:,:,k)=V(:,:,k)*xk;
          end % k
       end % K>1
    else % x is cell
       K=numel(x);
       sizex=size(x);
       V=cell(sizex);
       U=cell(sizex);
       z=cell(sizex);
       for k=1:K
          T=size(x{k},2);
          % Step 1. Center the data.
          xk=bsxfun(@minus,x{k},mean(x{k},2));

          % Step 2. Form MLE of data covariance.
          covar=xk*xk'/T;

          % Step 3. Eigen decomposition of covariance.
          [eigvec, eigval] = eig (covar);

          % Step 4. Forming whitening transformation.
          V{k}=sqrt(eigval) \ eigvec';
          U{k}=eigvec * sqrt(eigval);

          % Step 5. Form whitened data
          z{k}=V{k}*xk;
       end % k      
    end

end

