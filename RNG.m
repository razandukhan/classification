close all
clear all
% generating for classes with rand data  
D1=rand(100,3)+[2*ones(size(rand(100,1))) 2*ones(size(rand(100,1))) 2*ones(100,1)];
D2=rand(100,3)+[-2*ones(size(rand(100,1))) -2*ones(size(rand(100,1))) ones(100,1)];
D3=rand(10,3)+[2*ones(size(rand(10,1))) -2*ones(size(rand(10,1))) ones(10,1)];
D4=rand(10,3)+[-2*ones(size(rand(10,1))) 2*ones(size(rand(10,1))) 2*ones(10,1)];
% plotting this 2 classes rand data
plot(D1(:,1),D1(:,2),'k*')
hold on
plot(D2(:,1),D2(:,2),'K*')
plot(D3(:,1),D3(:,2),'r*')
plot(D4(:,1),D4(:,2),'r*')
axis([-10 10 -10 10])
% generating matrix include the past classes 
data=[D1;D2;D3;D4]
pr=0.5
%splitting the data into taining and testing by 0.5 splitting factor
%this trainingindices is a rand data with the same size of general data and
%refer to indexes of each element in the data
trainingindices=randi([1 size(data,1)],pr*round(size(data,1)),1);
trainingData=data(trainingindices,:);
data(trainingindices,:)=[];
testData=data;
%elm classifier with classification type and one hiddenneurons 
[trainingAccuracy,testingAccuracy,train,test] = ELM(trainingData,testData,1,1,'sig',2);
%FlN classifier
[train,test] = FLN(trainingData,testData,1,1,'sig',2)


