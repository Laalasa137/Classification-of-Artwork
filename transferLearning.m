%Uncomment the below section to compute ROC plot for Style classification 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%     Symmetry_Groups = {'$0$abstract_painting','$1$cityscape','$2$genre_painting','$3$illustration','$4$landscape',...					
%      '$5$nude_painting','$6$portrait','$7$religious_painting','$8$sketch_and_study','$9$still_life'};
%      train_folder = 'datagenretrain_old';
%     test_folder  = 'datagenreval_old';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




checkpointDir = 'modelCheckpoints';
resultsDir = 'resultsDir';


rng(1) % For reproducibility
                       
% uncomment this to do preprocessing from old wikiart dataset(after categorizing)
% train_folder = 'train_aug';
% test_folder  = 'test_aug';
% alexNetDataAug(dataDir, train_folder, test_folder, Symmetry_Groups, resultsDir)


fprintf('Loading Train Filenames and Label Data...'); 
t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
% Split with validation set
[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));
fprintf('Loading Test Filenames and Label Data...'); 
t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

%%
rng('default');
numEpochs = 1; % 5 for both learning rates
nTraining = length(train.Labels);

net = alexnet;
layersTransfer = net.Layers(1:end-3);

layers = [
    layersTransfer;
    %Change Last layer output classes size accordingly
    %23-artist,9-Genre,27 Style
    fullyConnectedLayer(23,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20);
    softmaxLayer();
    classificationLayer();
    ];

learning_rates = [1e-4];
batchSize = [128];

transferlearning = zeros(size(learning_rates,2)*size(batchSize,2),6);

for i=1:numel(batchSize)
    for j=1:numel(learning_rates)
      fprintf("Training with learning rate = %0.7f", learning_rates(1,j));
      if ~exist(checkpointDir,'dir'); 
        mkdir(checkpointDir); 
      end
      if ~exist(resultsDir,'dir'); 
        mkdir(resultsDir); 
      end
      % Set the training options
      options = trainingOptions('sgdm','MaxEpochs',20,... 
        'InitialLearnRate',learning_rates(1,j),...% learning rate
        'CheckpointPath', checkpointDir,...
        'MiniBatchSize', batchSize(1,i), ...
        'MaxEpochs',numEpochs);
    
      t = tic;
      [net1,info1] = trainNetwork(train,layers,options);
      fprintf('Trained in %.02f seconds\n', toc(t));
      save('net4.mat','net1');
      
      % Test on the validation data
      YTrain = classify(net1,train);
      train_acc = mean(YTrain==train.Labels)
      train_confmat = confusionmat(train.Labels, YTrain)
      clf
      f = figure(1);
      train_confchart = confusionchart(train.Labels, YTrain);
      set(f,'position',[0,0,1000,900]);
      fname = sprintf('resultsDir/train_confchart.png')
      saveas(f, fname);
      
      YVal = classify(net1,val);
      val_acc = mean(YVal==val.Labels)
      val_confmat = confusionmat(val.Labels, YVal)
      clf
      f = figure(2);
      val_confchart = confusionchart(val.Labels, YVal);
      set(f,'position',[0,0,1000,900]);
      fname = sprintf('resultsDir/val_confchart.png');
      saveas(f, fname);
        
      YTest = classify(net1,test);
      test_acc = mean(YTest==test.Labels)
      test_confmat = confusionmat(test.Labels, YTest)
      clf
      f = figure(3);
      test_confchart = confusionchart(test.Labels, YTest);
      set(f,'position',[0,0,1000,900]);
      fname = sprintf('resultsDir/test_confchart.png');
      saveas(f, fname);
      
      transferlearning(1,1) = learning_rates(1,j);
      transferlearning(1,2) = batchSize(1,i);
      transferlearning(1,3) = train_acc;
      transferlearning(1,4) = val_acc;
      transferlearning(1,5) = test_acc;
      transferlearning(1,6) = toc(t);
   
      clf
      f = figure(6); 
      plotTrainingAccuracy_All(info1,numEpochs);
      fname = sprintf('resultsDir/plotTrainingAccuracy.png');
      set(f,'position',[0,0,1000,900]);
      saveas(f, fname);
      rmdir('modelCheckpoints', 's');
      
    end
end
save('transferlearning4.mat', 'transferlearning');



