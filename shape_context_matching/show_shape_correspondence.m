% Shape Context
% MNIST digits

% load in the digit database (only needs to be done once per session)
if ~(exist('train_data')&exist('label_train'))
   %load digit_100_train_easy;
  
  load digit_100_train_hard;
end

% choose which two digits to compare:
mm=49;
nn=27;

%%%
%%%Define flags and parameters:
%%%
display_flag=1;
affine_start_flag=1;
polarity_flag=1;
nsamp=100;
eps_dum=0.25;
ndum_frac=0.25;        
mean_dist_global=[];
ori_weight=0.1;
nbins_theta=12;
nbins_r=5;
r_inner=1/8;
r_outer=2;
tan_eps=1.0;
n_iter=6;
beta_init=1;
r=1; % annealing rate
w=4;
sf=2.5;

cmap=flipud(gray);

%%%
%%% image loading
%%%
V1=reshape(train_data(mm,:),28,28)';
#V1=imresize(V1,sf,'bil');
V2=reshape(train_data(nn,:),28,28)';
#V2=imresize(V2,sf,'bil');
[N1,N2]=size(V1);

if display_flag
   figure(1)
   subplot(2,2,1)
   imagesc(V1);axis('image')
   title(int2str(mm))
   subplot(2,2,2)
   imagesc(V2);axis('image')
   title(int2str(nn))
   colormap(cmap)
   drawnow
end

%%%
%%% edge detection
%%%
[x2,y2,t2]=bdry_extract_3(V2);
nsamp2=length(x2)
if nsamp2>=nsamp
   [x2,y2,t2]=get_samples_1(x2,y2,t2,nsamp);
else
   error('shape #2 doesn''t have enough samples')
end
Y=[x2 y2];

% get boundary points
disp('extracting boundary points...')
[x1,y1,t1]=bdry_extract_3(V1);

nsamp1=length(x1);
if nsamp1>=nsamp
   [x1,y1,t1]=get_samples_1(x1,y1,t1,nsamp);
else
   error('shape #1 doesn''t have enough samples')
end
X=[x1 y1];

if display_flag
   subplot(2,2,3)
   plot(X(:,1),X(:,2),'b+')
   hold on
   quiver(X(:,1),X(:,2),cos(t1),sin(t1),0.5,'b')
   hold off
   axis('ij');axis([1 N2 1 N1])
   title([int2str(length(x1)) ' samples'])
   subplot(2,2,4)
   plot(Y(:,1),Y(:,2),'ro')
   hold on
   quiver(Y(:,1),Y(:,2),cos(t2),sin(t2),0.5,'r.')
   hold off
   axis('ij');axis([1 N2 1 N1])
   title([int2str(length(x2)) ' samples'])
   drawnow	
end

if display_flag
   [x,y]=meshgrid(linspace(1,N2,36),linspace(1,N1,36));
   x=x(:);y=y(:);M=length(x);
end

%%%
%%% compute correspondences
%%%
Xk=X;
tk=t1;
k=1;
s=1;
ndum=round(ndum_frac*nsamp);
out_vec_1=zeros(1,nsamp);
out_vec_2=zeros(1,nsamp);
while s
   disp(['iter=' int2str(k)])
   disp('computing shape contexts for (deformed) model...')
   [BH1,mean_dist_1]=sc_compute(Xk',zeros(1,nsamp),mean_dist_global,nbins_theta,nbins_r,r_inner,r_outer,out_vec_1);
   disp('done.')
   % apply the scale estimate from the warped model to the test shape
   disp('computing shape contexts for target...')
   [BH2,mean_dist_2]=sc_compute(Y',zeros(1,nsamp),mean_dist_global,nbins_theta,nbins_r,r_inner,r_outer,out_vec_2);
   disp('done.')

   if affine_start_flag
      if k==1
	 % use huge regularization to get affine behavior
	 lambda_o=1000;
      else
	 lambda_o=beta_init*r^(k-2);	 
      end
   else
      lambda_o=beta_init*r^(k-1);
   end
   beta_k=(mean_dist_2^2)*lambda_o;

   costmat_shape=hist_cost_2(BH1,BH2);
   theta_diff=repmat(tk,1,nsamp)-repmat(t2',nsamp,1);
%   costmat_theta=abs(atan2(sin(theta_diff),cos(theta_diff)))/pi;
   if polarity_flag
      % use edge polarity
      costmat_theta=0.5*(1-cos(theta_diff));
   else
      % ignore edge polarity
      costmat_theta=0.5*(1-cos(2*theta_diff));
   end      
   costmat=(1-ori_weight)*costmat_shape+ori_weight*costmat_theta;
   nptsd=nsamp+ndum;
   costmat2=eps_dum*ones(nptsd,nptsd);
   costmat2(1:nsamp,1:nsamp)=costmat;
   cvec=hungarian(costmat2);
%   cvec=hungarian_fast(costmat2);

   % update outlier indicator vectors
   [a,cvec2]=sort(cvec);
   out_vec_1=cvec2(1:nsamp)>nsamp;
   out_vec_2=cvec(1:nsamp)>nsamp;

   X2=NaN*ones(nptsd,2);
   X2(1:nsamp,:)=Xk;
   X2=X2(cvec,:);
   X2b=NaN*ones(nptsd,2);
   X2b(1:nsamp,:)=X;
   X2b=X2b(cvec,:);
   Y2=NaN*ones(nptsd,2);
   Y2(1:nsamp,:)=Y;

   % extract coordinates of non-dummy correspondences and use them
   % to estimate transformation
   ind_good=find(~isnan(X2b(1:nsamp,1)));
   n_good=length(ind_good);
   X3b=X2b(ind_good,:);
   Y3=Y2(ind_good,:);

   if display_flag
      figure(2)
      plot(X2(:,1),X2(:,2),'b+',Y2(:,1),Y2(:,2),'ro')
      hold on
      h=plot([X2(:,1) Y2(:,1)]',[X2(:,2) Y2(:,2)]','k-');
      
      if 1
%	 set(h,'linewidth',1)
	 quiver(Xk(:,1),Xk(:,2),cos(tk),sin(tk),0.5,'b')
	 quiver(Y(:,1),Y(:,2),cos(t2),sin(t2),0.5,'r')
      end
      hold off
      axis('ij')
      title([int2str(n_good) ' correspondences (warped X)'])
      axis([1 N2 1 N1])
      drawnow	
   end
   
   if display_flag
      % show the correspondences between the untransformed images
      figure(3)
      plot(X(:,1),X(:,2),'b+',Y(:,1),Y(:,2),'ro')
      ind=cvec(ind_good);
      hold on
      plot([X2b(:,1) Y2(:,1)]',[X2b(:,2) Y2(:,2)]','k-')
      hold off
      axis('ij')
      title([int2str(n_good) ' correspondences (unwarped X)'])
      axis([1 N2 1 N1])
      drawnow	
   end

   [cx,cy,E]=bookstein(X3b,Y3,beta_k);

   % calculate affine cost
   A=[cx(n_good+2:n_good+3,:) cy(n_good+2:n_good+3,:)];
   s=svd(A);
   aff_cost=log(s(1)/s(2));
   
   % calculate shape context cost
   [a1,b1]=min(costmat,[],1);
   [a2,b2]=min(costmat,[],2);
   sc_cost=max(mean(a1),mean(a2));
   
   % warp each coordinate
   fx_aff=cx(n_good+1:n_good+3)'*[ones(1,nsamp); X'];
   d2=max(dist2(X3b,X),0);
   U=d2.*log(d2+eps);
   fx_wrp=cx(1:n_good)'*U;
   fx=fx_aff+fx_wrp;
   fy_aff=cy(n_good+1:n_good+3)'*[ones(1,nsamp); X'];
   fy_wrp=cy(1:n_good)'*U;
   fy=fy_aff+fy_wrp;

   Z=[fx; fy]';

   % apply the warp to the tangent vectors to get the new angles
   Xtan=X+tan_eps*[cos(t1) sin(t1)];
   fx_aff=cx(n_good+1:n_good+3)'*[ones(1,nsamp); Xtan'];
   d2=max(dist2(X3b,Xtan),0);
   U=d2.*log(d2+eps);
   fx_wrp=cx(1:n_good)'*U;
   fx=fx_aff+fx_wrp;
   fy_aff=cy(n_good+1:n_good+3)'*[ones(1,nsamp); Xtan'];
   fy_wrp=cy(1:n_good)'*U;
   fy=fy_aff+fy_wrp;
   
   Ztan=[fx; fy]';
   tk=atan2(Ztan(:,2)-Z(:,2),Ztan(:,1)-Z(:,1));
  %{
   if display_flag
      figure(4)
      plot(Z(:,1),Z(:,2),'b+',Y(:,1),Y(:,2),'ro');
      axis('ij')
      title(['k=' int2str(k) ', \lambda_o=' num2str(lambda_o) ', I_f=' num2str(E) ', aff.cost=' num2str(aff_cost) ', SC cost=' num2str(sc_cost)])
      axis([1 N2 1 N1])
      % show warped coordinate grid
      fx_aff=cx(n_good+1:n_good+3)'*[ones(1,M); x'; y'];
      d2=dist2(X3b,[x y]);
      fx_wrp=cx(1:n_good)'*(d2.*log(d2+eps));
      fx=fx_aff+fx_wrp;
      fy_aff=cy(n_good+1:n_good+3)'*[ones(1,M); x'; y'];
      fy_wrp=cy(1:n_good)'*(d2.*log(d2+eps));
      fy=fy_aff+fy_wrp;
      hold on
      plot(fx,fy,'k.','markersize',1)
      hold off
      drawnow
   end
  %} 
   % update Xk for the next iteration
   Xk=Z;
  
   if k==n_iter
      s=0;
   else
      k=k+1;
   end
end