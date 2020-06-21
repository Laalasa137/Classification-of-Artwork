function visualization()
%Uncomment the below section to compute ROC plot for Style classification 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %alex_net = load('./net4.mat');
  %dataDir= './processed_Data/Style/';
  %   Symmetry_Groups = {'$0$Abstract_Expressionism','$1$Action_painting','$2$Analytical_Cubism','$3$Art_Nouveau','$4$Baroque',...
  % 					'$5$Color_Field_Painting','$6$Contemporary_Realism','$7$Cubism','$8$Early_Renaissance','$9$Expressionism','$10$Fauvism','$11$High_Renaissance',...
  % 					'$12$Impressionism','$13$Mannerism_Late_Renaissance','$14$Minimalism','$15$Naive_Art_Primitivism','$16$New_Realism','$17$Northern_Renaissance','$18$Pointillism',...
  %                     '$19$Pop_Art','$20$Post_Impressionism','$21$Realism','$22$Rococo','$23$Romanticism','$24$Symbolism','$25$Synthetic_Cubism','$26$Ukiyo_e'};
  %   
  %train_folder = 'datastyletrain_old';
  %test_folder  = 'datastyleval_old';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
%Uncomment the below section to compute ROC plot for Artist classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  alex_net = load('./net2.mat');
  dataDir= './processed_Data/Artist/';
    Symmetry_Groups = {'$0$Albrecht_Durer','$1$Boris_Kustodiev','$2$Camille_Pissarro','$3$Childe_Hassam','$4$Claude_Monet',...
  					'$5$Edgar_Degas','$6$Eugene_Boudin','$7$Gustave_Dore','$8$Ilya_Repin','$9$Ivan_Aivazovsky','$10$Ivan_Shishkin','$11$John_Singer_Sargent',...
  					'$12$Marc_Chagall','$13$Martiros_Saryan','$14$Nicholas_Roerich','$15$Pablo_Picasso','$16$Paul_Cezanne','$17$Pierre_Auguste_Renoir','$18$Pyotr_Konchalovsky',...
                      '$19$Raphael_Kirchner','$20$Rembrandt','$21$Salvador_Dali','$22$Vincent_van_Gogh'};
  
   train_folder = 'dataartisttrain_old';
   test_folder  = 'dataartistval_old';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Uncomment the below section to compute ROC plot for Genre classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%   dataDir= './processed_Data/Genre/';
%   alex_net = load('./net3.mat');
%     Symmetry_Groups = {'$0$abstract_painting','$1$cityscape','$2$genre_painting','$3$illustration','$4$landscape',...					
%      '$5$nude_painting','$6$portrait','$7$religious_painting','$8$sketch_and_study','$9$still_life'};
%      train_folder = 'datagenretrain_old';
%     test_folder  = 'datagenreval_old';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
fname = "alex_original_filter";
net = alex_net;
conv_layer=17;
                     
train_folder = 'dataartisttrain_old';
test_folder  = 'dataartistval_old';
fprintf('Loading Train Filenames and Label Data...'); 
t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
% Split with validation set
[train, val] = splitEachLabel(train_all,.9);
t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t))



%analyzeNetwork(net.net1)
%%%%%%%%%%%%%%%%%%%%%%%%FILTER VISUALIZATION%%%%%%%%%%%%%%%%%%%%%%%%
%Change Numbers in p and q accordingly to visualize 96=3*4*8 (4*8) layers
p = 4;
q = 8;
%For Visualizing First Layer Filters Across all networks
conv_layer = 14;
filterCount = net.net1.Layers(conv_layer).NumFiltersPerGroup
layer_name = net.net1.Layers(conv_layer).Name
channels = 1:filterCount;
I = deepDreamImage(net.net1,conv_layer,channels);
figure(1);
title(['Layer ','layer_name','Features'])
clf
cnt = 3; %changing cnt accordingly
f = figure(1);
channels = 1:32; 
j = 1 ; %change j accordingly 1:32, 33:64, 65:96
for i = channels
 set(f,'position',[0,0,1000,900]);
 subplot(p,q,i)
 imshow(I(:,:,:,j))
 j=j+1;
end














%%%%%%%%%%%%%%%%%%%%TSNE Visualization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UNCOMMENT training, testing, validation parts to view tsne accordingly 
%%%%%%%%%%%%%%%%% N - Dimesnions%%%%%%%%%%%%%%%%%%%%%%
% 
% featuresTrain = activations(net.net1,train,conv_layer,'OutputAs','rows');
% Y = tsne(featuresTrain,'Algorithm','barneshut','NumPCAComponents',10);
% f = figure(1);
% gscatter(Y(:,1), Y(:,2), train.Labels);
% title('gscatter');
% set(f,'position',[0,0,1000,900]);
% cname = sprintf('resultsDir/%s_artistttttt_train_sne.png', fname);
% saveas(f, cname);
% fprintf("First Figure Done =%.02f", toc(t))

% %%%%%%%%%%%%%%%%For Validation%%%%%%%%%%%%%%%%%%%%%%%%%%%
% featuresTrain = activations(net.net1,val,conv_layer,'OutputAs','rows');
% Y = tsne(featuresTrain,'Algorithm','barneshut','NumPCAComponents',10);
% f = figure(2);
% gscatter(Y(:,1), Y(:,2), val.Labels);
% title('gscatter');
% set(f,'position',[0,0,1000,900]);
% dname = sprintf('resultsDir/%s_artistttttttt_val_sne.png', fname);
% saveas(f, dname);
% fprintf("Second Figure Done =%.02f", toc(t));
% %%%%%%%%%%%%%%For Testing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% featuresTrain = activations(net.net1,test,conv_layer,'OutputAs','rows');
% Y = tsne(featuresTrain,'Algorithm','barneshut','NumPCAComponents',10);
% f = figure(3);
% gscatter(Y(:,1), Y(:,2), test.Labels);
% title('gscatter');
% set(gcf,'position',[0,0,1000,900]);
% ename = sprintf('resultsDir/%s_artisttttttt_test_sne.png', fname);
% saveas(f, ename);
% fprintf("Third Figure Done =%.02f", toc(t));
% 

%%%%%%%%%%%%%%%%%3 - Dimesnions%%%%%%%%%%%%%%%%%%%%%%
% 
% featuresTrain = activations(net.net1,train,conv_layer,'OutputAs','rows');
% Y = tsne(featuresTrain,'Algorithm','barneshut','NumDimensions',3);
% f = figure(1);
% v = double(categorical(train.Labels));
% c = full(sparse(1:numel(v),v,ones(size(v)),numel(v),3));
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled')
% title('3-D Embedding')
% view(-50,8)
% % gscatter(Y(:,1), Y(:,2), train.Labels);
% % title('gscatter');
% set(f,'position',[0,0,1000,900]);
% cname = sprintf('resultsDir/%s_train3_artist_sne.png', fname);
% saveas(f, cname);
% fprintf("First Figure Done =%.02f", toc(t))
% 
% %%%%%%%%%%%%%%%%For Validation%%%%%%%%%%%%%%%%%%%%%%%%%%%
% featuresTrain = activations(net.net1,val,conv_layer,'OutputAs','rows');
% Y = tsne(featuresTrain,'Algorithm','barneshut','NumDimensions',3);
% f=figure(2);
% v = double(categorical(val.Labels));
% c = full(sparse(1:numel(v),v,ones(size(v)),numel(v),3));
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled')
% title('3-D Embedding')
% view(-50,8)
% % gscatter(Y(:,1), Y(:,2), train.Labels);
% % title('gscatter');
% %title('gscatter');
% set(f,'position',[0,0,1000,900]);
% dname = sprintf('resultsDir/%s_artist3_val_sne.png', fname);
% saveas(f, dname);
% fprintf("Second Figure Done =%.02f", toc(t));
% 
% 
% %%%%%%%%%%%%%%For Testing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% featuresTrain = activations(net.net1,test,conv_layer,'OutputAs','rows');
% Y = tsne(featuresTrain,'Algorithm','barneshut','NumDimensions',3);
% f = figure(3);
% v = double(categorical(test.Labels));
% c = full(sparse(1:numel(v),v,ones(size(v)),numel(v),3));
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled')
% title('3-D Embedding')
% view(-50,8)
% % gscatter(Y(:,1), Y(:,2), train.Labels);
% % title('gscatter');
% set(gcf,'position',[0,0,1000,900]);
% ename = sprintf('resultsDir/%s_artist3_test_sne.png', fname);
% saveas(f, ename);
% fprintf("Third Figure Done =%.02f", toc(t));
% 
% 
% 
% 
% 
% 
end














































































% im = imread('./Image_2.jpg');
% imshow(im)
% alex = load('./net3.mat');
% imgSize = size(im);
% imgSize = imgSize(1:2)
% analyzeNetwork(alex.net1)
% act1 = activations(alex.net1,im,'conv1');
% sz = size(act1);
% act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
% I = imtile(mat2gray(act1),'GridSize',[4 4]);
% imshow(I)
% [maxValue,maxValueIndex] = max(max(max(act1)));
% act1chMax = act1(:,:,:,maxValueIndex);
% act1chMax = mat2gray(act1chMax);
% act1chMax = imresize(act1chMax,imgSize);
% I = imtile({im,act1chMax});
% imshow(I)
% layer = 23;
% channels = [1 2 3 4 5 6 7 8 9 10];
% name = alex.net1.Layers(layer).Name
% alex.net1.Layers(end).Classes(channels)
% I = deepDreamImage(alex.net1,name,channels, ...
%     'Verbose',false, ...
%     'NumIterations',100, ...
%     'PyramidLevels',2);
% figure
% I = imtile(I,'ThumbnailSize',[250 250]);
% imshow(I)
% name = net.Layers(layer).Name;
% title(['Layer ',name,' Features'])













% fname = "alex_orignal_filter";
% net = alex;
% %Change Numbers in p and q accordingly to visualize 96=3*4*8 (4*8) layers
% p = 4;
% q = 8;
% %For Visualizing First Layer Filters Across all networks
% conv_layer = 2;
% filterCount = net.net1.Layers(conv_layer).NumFilters; 
% layer_name = net.net1.Layers(conv_layer).Name;
% channels = 1:filterCount;
% I = deepDreamImage(net.net1,conv_layer,channels);
% figure(1);
% title(['Layer ','layer_name','Features'])
% clf
% cnt = 3; %changing cnt accordingly
% f = figure(1);
% channels = 1:32; 
% j = 65 ; %change j accordingly 1:32, 33:64, 65:96
% for i = channels
%  set(f,'position',[0,0,1000,900]);
%  subplot(p,q,i)
%  imshow(I(:,:,:,j))
%  j=j+1;
% end
% fname = sprintf('./resultsDir/%s_layer%d.png', fname, cnt);
% saveas(f, fname);