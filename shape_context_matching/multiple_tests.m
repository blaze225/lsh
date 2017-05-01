warning('off','Matlab-style short-circuit operation performed for operator &');
clear all;

display_flag=1;
cmap=flipud(gray);

% load in the digit database (only needs to be done once per session)
if ~(exist('train_data')&exist('label_train'))
   %load digit_100_train_easy;
  
  load digit_100_train_hard;
end

%cost_final=zeros(1,10);
%cost_final=containers.Map;
indices=zeros(1,100);
values=zeros(1,100);
%s = struct ();
query_image=15;
V2=reshape(train_data(query_image,:),28,28)';

if display_flag
   figure(1)
   subplot(2,2,1)
   imagesc(V2);axis('image')
   title(int2str(query_image))
   colormap(cmap)
   drawnow
end
for i=1:100

  V1=reshape(train_data(i,:),28,28)';
  
  %cost_final(i/5)=calculate_cost(V1,V2);
  %cost_final(i/5)
  %s = setfield (s, i, calculate_cost(V1,V2));
  data.(num2str(calculate_cost(V1,V2)))=i;
  indices(i)=i;
  values(i)=calculate_cost(V1,V2);
end

%cost_final
data
indices
values

len=length(values)
for i=1:len-1
  for j=i+1:len
    if(values(i)>values(j))
      temp=values(i);
      values(i)=values(j);
      values(j)=temp;
      
      temp=indices(i);
      indices(i)=indices(j);
      indices(j)=temp;
    end
  end
end

indices
values

matched_1=reshape(train_data(indices(1),:),28,28)';
matched_2=reshape(train_data(indices(2),:),28,28)';
matched_3=reshape(train_data(indices(3),:),28,28)';
matched_4=reshape(train_data(indices(4),:),28,28)';
%V1=imresize(V1,sf,'bil');
V2=reshape(train_data(query_image,:),28,28)';
%V2=imresize(V2,sf,'bil');
[N1,N2]=size(V1);

if display_flag
   figure(2)
   subplot(2,2,1)
   imagesc(matched_1);axis('image')
   title([int2str(indices(1)) ' with SDD ' num2str(values(1))])
   subplot(2,2,2)
   %imagesc(V2);axis('image')
   %title(int2str(query_image))
   imagesc(matched_2);axis('image')
   title([int2str(indices(2)) ' with SDD ' num2str(values(2))])
   subplot(2,2,3)
   imagesc(matched_3);axis('image')
   title([int2str(indices(3)) ' with SDD ' num2str(values(3))])
   subplot(2,2,4)
   imagesc(matched_4);axis('image')
   title([int2str(indices(4)) ' with SDD ' num2str(values(4))])
   colormap(cmap)
   drawnow
end