function [train,test] = FLN(trainingData,testingData,FlnType,numberOfHiddenNeurons,activationFunctionType,numberOfClasses)

% Usage: [train,test] = FLN(trainingData,testingData,FlnType,numberOfHiddenNeurons,activationFunctionType,numberOfClasses)
%------------------------------------------------------------------------------------------------------------
% Input:
%-------
% trainingData              - training dataset
% testingData               - testing dataset
% FlnType                   - 0 for regression, 1 for (both binary and multi-classes) classification
% numberOfHiddenNeurons     - Number of hidden neurons assigned to the FLN
% activationFunctionType    - Type of activation function:
%                               'sig' for Sigmoidal function
%                               'sin' for Sine function
%                               'hardlim' for Hardlim function
%                               'tribas' for Triangular basis function
%                               'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
% numberOfClasses           - Number of required classes to which the data will be classified
%------------------------------------------------------------------------------------------------------------
% Output:
%--------
% train                     - Structure contains the class-label column
%                             (targets) and the actual output of classifier
%                             (outputs) for training dataset
% test                      - Structure contains the class-label column
%                             (targets) and the actual output of classifier
%                             (outputs) for testing dataset
%------------------------------------------------------------------------------------------------------------
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output neurons
% (for example, if neuron 5 has the highest output means input belongs to 5-th class)
%------------------------------------------------------------------------------------------------------------

%% Macro definition

REGRESSION = 0;
CLASSIFIER = 1;

%% Preparing datasets

% Target data for training dataset (1×N where: N is the No. of records in training dataset)
trainDat.Target = trainingData(:,size(trainingData,2))';
% Input data for training dataset (n×N where: n is length of the input)
trainDat.Input = trainingData(:,1:size(trainingData,2)-1)';

% Target data for testing dataset (1×N1 where: N1 is the No. of records in testing dataset)
testDat.Target = testingData(:,size(testingData,2))';
% Input data for testing dataset (n×N1)
testDat.Input = testingData(:,1:size(testingData,2)-1)';

numberOfTrainingData = size(trainDat.Input,2); % N
numberOfTestingData = size(testDat.Input,2);   % N1
numberOfInputNeurons = size(trainDat.Input,1); % n (number of features)

%% Processing the targets of training dataset

if FlnType==CLASSIFIER
    tempTrain = zeros(numberOfClasses,numberOfTrainingData);
    ind = sub2ind(size(tempTrain),trainDat.Target,1:numberOfTrainingData);
    tempTrain(ind) = 1;
    trainDat.ProcessedTarget = tempTrain*2-1;
end

%% Calculate the hidden output-weight matrix (G) and matrix (H)

% Random generating of the input weights matrix (w_in) and the biases of hidden neurons (b)
W_in = rand(numberOfHiddenNeurons,numberOfInputNeurons)*2-1; % (m×n where: m is No. of hidden neurons, n is No. of input neurons)
biasOfHiddenNeurons = rand(numberOfHiddenNeurons,1);  % m×1

%%%% For training dataset %%%%

% Calculate matrix G
biasMatrix = repmat(biasOfHiddenNeurons,1,numberOfTrainingData); % m×N
tempG_train = W_in*trainDat.Input+biasMatrix; % m×N
G_train = ActivationFunction(tempG_train,activationFunctionType); % m×N

% Calculate matrix H (H = [X ; G ; I])
H_train = [trainDat.Input ; G_train ; ones(1,numberOfTrainingData)];  % n+m+1×N

%%%% For testing dataset %%%%

% Calculate matrix G
biasMatrix = repmat(biasOfHiddenNeurons,1,numberOfTestingData);  % m×N1
tempG_test = W_in*testDat.Input+biasMatrix;   % m×N1
G_test = ActivationFunction(tempG_test,activationFunctionType); % m×N1

% Calculate matrix H
H_test = [testDat.Input ; G_test ; ones(1,numberOfTestingData)];  % n+m+1×N1

%% Calculate the connection matrix W

% Linear activation function in the output layer
if FlnType==CLASSIFIER
    W = trainDat.ProcessedTarget*pinv(H_train);  % C×n+m+1
elseif FlnType==REGRESSION
    W = trainDat.Target*pinv(H_train);
end

%% Calculating the actual output of training and testing data

ActualOutputOfTrainData = W*H_train;
ActualOutputOfTestData = W*H_test;

%% Outputs and Targets

train.targets = trainDat.Target;
test.targets = testDat.Target;
if FlnType==CLASSIFIER
    [~,labelIndexActualTrain] = max(ActualOutputOfTrainData);
    train.outputs = labelIndexActualTrain;
    [~,labelIndexActualTest] = max(ActualOutputOfTestData);
    test.outputs = labelIndexActualTest;
elseif FlnType==REGRESSION
    train.outputs = ActualOutputOfTrainData;
    test.outputs = ActualOutputOfTestData;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function H = ActivationFunction(H_temp,activationFunctionType)

switch lower(activationFunctionType)
    case {'sig','sigmoid'} % Sigmoid
        H = 1./(1 + exp(-H_temp));
    case {'sin','sine'} % Sine
        H = sin(H_temp);
    case {'hardlim'}  % Hard Limit
        H = double(hardlim(H_temp));
    case {'tribas'}  % Triangular basis function
        H = tribas(H_temp);
    case {'radbas'}  % Radial basis function
        H = radbas(H_temp);
end

end