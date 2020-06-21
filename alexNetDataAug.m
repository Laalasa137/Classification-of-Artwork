function alexNetDataAug(oldpath, train_folder, test_folder, Symmetry_Groups, resultsDir)

%Augmentation for generating for passing into alex net
%involves scale image to 227X227 and into RGB Scale
alx = '_old';

%Uncomment and change the folders correspondingly to generate data
%source_train_folder = strcat(oldpath,train_folder);
%destination_train_folder = strcat(oldpath,train_folder,alx);
source_train_folder = strcat(oldpath,test_folder);
destination_train_folder = strcat(oldpath,test_folder,alx);
train_all = imageDatastore(fullfile(source_train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
if ~exist(destination_train_folder,'dir')
    for i = 1: length(Symmetry_Groups)
        dir = char(Symmetry_Groups(i));
        mkdir(destination_train_folder, dir);
    end
end
total_train_images = size(train_all.Labels);
combinations = 1;
t = tic;
    for alpha=1:total_train_images
        [image, fileinfo] = readimage(train_all, alpha);
        fname = split(fileinfo.Filename, '/');
        fname;
        for beta=1:combinations
            chr = char(fname(length(fname)));
            lastdot_pos = find(chr == '.', 1, 'last');
            part12 = chr(1 : lastdot_pos - 1);
            newimagename = strcat(part12,'.jpg');
            newimagename=char(fname);
            parts = strsplit(newimagename, '\');
            folder = char(parts(1,size(parts,2)-1));
            filename = char(parts(1,size(parts,2)-0));
            final_fname = strcat(destination_train_folder, '/' , folder, '/', filename);
            transformed_Image = imresize(image,[227 227]);
            %size(transformed_Image)
            %transformed_Image_rgb = transformed_Image(:,:,[1 1 1]);
            %transformed_Image_rgb = cat(3, transformed_Image, transformed_Image, transformed_Image);
            %size(transformed_Image_rgb)
            final_fname;
            imwrite(transformed_Image,final_fname);
        end
        if(mod(alpha, 100)==0)
            fprintf('Time Elapsed is %.02f seconds and alpha=%d\n', toc(t), alpha);
        end
    end
end