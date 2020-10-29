tic;
nameImageFile='50testcoba';
file = '50testcoba2.xlsx';

newImage = imageDatastore(fullfile(nameImageFile));
ds = augmentedImageDatastore(imageSize, newImage,...
    'ColorPreprocessing', 'gray2rgb');
imageFeatures = activations(net, ds, featureLayer,...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');

searchImage=dir(nameImageFile);

nameImage=[];
for i=3:length(searchImage)
    nameImage=[nameImage;string(searchImage(i).name)];
end

solveName=nameImage;
solveCategories=flip(string(rot90(categories(label))));

solvePrediction=[solveName solveCategories];
xlswrite(file,solvePrediction);

time = toc;
sprintf('Finished in %f seconds',time)