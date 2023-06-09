% Load and preprocess the dataset
imageFolder = 'C:\Users\Elham\Desktop\gemstoneTest';
imds = imageDatastore(imageFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7);

imdsTrain.ReadFcn = @(imdsTrain)imresize(imread(imdsTrain), [224, 224]);
imdsTest.ReadFcn = @(imdsTest)imresize(imread(imdsTest), [224, 224]);

% Create MobileNet-v2 network
net = mobilenetv2;

% Modify the fully connected layer for the number of classes in your dataset
numClasses = numel(categories(imdsTrain.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
lgraph = replaceLayer(lgraph, 'Logits', newFCLayer);
newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_Logits', newClassLayer);

% Define training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 8, ...
    'MiniBatchSize', 11, ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 5, ...
    'Verbose', true);

% Train the network
trainedNet = trainNetwork(imdsTrain, lgraph, options);

predictions = classify(trainedNet, imdsTest);
plotconfusion(imdsTest.Labels, predictions);

