function ROCPlots()
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
  
%   alex_net = load('./net2.mat');
%   dataDir= './processed_Data/Artist/';
%     Symmetry_Groups = {'$0$Albrecht_Durer','$1$Boris_Kustodiev','$2$Camille_Pissarro','$3$Childe_Hassam','$4$Claude_Monet',...
%   					'$5$Edgar_Degas','$6$Eugene_Boudin','$7$Gustave_Dore','$8$Ilya_Repin','$9$Ivan_Aivazovsky','$10$Ivan_Shishkin','$11$John_Singer_Sargent',...
%   					'$12$Marc_Chagall','$13$Martiros_Saryan','$14$Nicholas_Roerich','$15$Pablo_Picasso','$16$Paul_Cezanne','$17$Pierre_Auguste_Renoir','$18$Pyotr_Konchalovsky',...
%                       '$19$Raphael_Kirchner','$20$Rembrandt','$21$Salvador_Dali','$22$Vincent_van_Gogh'};
%   
%    train_folder = 'dataartisttrain_old';
%    test_folder  = 'dataartistval_old';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Uncomment the below section to compute ROC plot for Genre classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
  dataDir= './processed_Data/Genre/';
  alex_net = load('./net3.mat');
    Symmetry_Groups = {'$0$abstract_painting','$1$cityscape','$2$genre_painting','$3$illustration','$4$landscape',...					
     '$5$nude_painting','$6$portrait','$7$religious_painting','$8$sketch_and_study','$9$still_life'};
     train_folder = 'datagenretrain_old';
    test_folder  = 'datagenreval_old';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
  net1 = alex_net.net1;
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
   
  [Ytest, scores] = classify(net1, test);

  f = figure();
  x = linspace(0,1);
  y = linspace(0,1);
  plot(x,y)
  hold on;
  for k=1:length(Symmetry_Groups)
    curstate=Symmetry_Groups{k};
    [X1, Y1] = perfcurve(test.Labels, scores(:,k),curstate);
    plot(X1,Y1)
%Uncomment the below section to compute ROC plot for Artist classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%     legend('Albrecht Durer','Boris Kustodiev','Camille Pissarro','Childe Hassam','Claude Monet',...
%             'Edgar Degas','Eugene Boudin','Gustave Dore','Ilya Repin','Ivan Aivazovsky','Ivan Shishkin','John Singer Sargent',...
%             'Marc Chagall','Martiros Saryan','Nicholas Roerich','Pablo Picasso','Paul Cezanne','Pierre Auguste Renoir','Pyotr Konchalovsky',...
%             'Raphael Kirchner','Rembrandt','Salvador Dali','Vincent van Gogh')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Uncomment the below section to compute ROC plot for Genre classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    legend('abstract painting','cityscape','genre painting','illustration','landscape','nude painting',...
       'portrait','religious painting','sketch and study','still life')  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Uncomment the below section to compute ROC plot for Style classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %     legend('Abstract Expressionism','Action painting','Analytical Cubism','Art Nouveau','Baroque',...
% 					'Color Field Painting','Contemporary Realism','Cubism','Early Renaissance','Expressionism','Fauvism','High Renaissance',...
% 					'Impressionism','Mannerism Late Renaissance','Minimalism','Naive Art Primitivism','New Realism','Northern Renaissance','Pointillism',...
%                     'Pop Art','Post Impressionism','Realism','Rococo','Romanticism','Symbolism','Synthetic Cubism','Ukiyo e')
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     pause(1)
  end
  
  xlabel('False positive rate') 
  ylabel('True positive rate')
  title('ROC Curve for Classification')
