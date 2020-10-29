tic;
rootFolder = fullfile('50train');
categories = {'00','01','02','03','04','05','06','07','08','09','10',...
    '11','12','13','14','15','16','17','18','19','20','21','22','23',...
    '24','25','26','27','28','29','30','31','32','33','34','35','36',...
    '37','38','39','40','41'};
imds = imageDatastore(fullfile(rootFolder,categories), 'LabelSource', 'foldername');

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

imds = splitEachLabel(imds,minSetCount, 'randomize');

net = resnet50();

[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

imageSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet,...
    'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet,...
    'ColorPreprocessing', 'gray2rgb');

featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer,...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLabels, 'Learner',...
    'Linear','Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures = activations(net, augmentedTestSet, featureLayer,...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

testLabels = testSet.Labels;
confMat = confusionmat(testLabels, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat,2));
akurasi = mean(diag(confMat));

time = toc;
sprintf('Finished in %f seconds',time)