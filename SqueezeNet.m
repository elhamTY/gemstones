% Load and preprocess the dataset
imageFolder = 'C:\Users\Elham\Desktop\gemstoneTest';
imds = imageDatastore(imageFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7);

inputSize = [227, 227, 3]; % Input image size expected by SqueezeNet

% Create image data augmenter
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

% Apply augmenter to image datastores
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

% Create SqueezeNet network
net = squeezenet;

% Modify the fully connected layer for the number of classes in your dataset
numClasses = numel(categories(imdsTrain.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
lgraph = replaceLayer(lgraph, 'relu_conv10', newFCLayer);
newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);

% Define training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 8, ...
    'MiniBatchSize', 11, ...
    'ValidationData', augimdsTest, ...
    'ValidationFrequency', 5, ...
    'Verbose', true);

% Train the network
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

% Evaluate the trained network on the test set
predictions = classify(trainedNet, augimdsTest);
plotconfusion(imdsTest.Labels, predictions);


