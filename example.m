% This is an example of Particle Competition and Cooperation (PCC) method 
% applied to the Wine Data Set from the UCI Machine Learning Repository
% Available at: https://archive.ics.uci.edu/ml/datasets/wine
%
% by Fabricio Breve - 21/01/2019
%
% Loading the Wine Data Set
load wine.data
% Getting the dataset attributes (all colums, except the first one).
X = wine(:,2:end);
% Getting dataset labels (first column). Labels should be >0 and in
% sequence. Ex.: 1, 2, 3.
label = wine(:,1);
% Randomly selecting 10% of the labels to be presented to the algorithm.
% An unlabeled item is represented by 0.
slabel = slabelgen(label,0.1);
disp('Running the algorithm MEX implementation, which is ~10 times faster than the pure Matlab implementation.');
disp('Parameters k: 10, distance: Normalized Euclidean, others: Default.');
tStart = tic;
owner = pccmex(X, slabel, 10, 'seuclidean');
tElapsed = toc(tStart);
% Evaluating the classification accuracy.
acc = stmwevalk(label,slabel,owner);
fprintf('Classification accuracy: %0.4f - Execution Time: %0.4fs\n\n',acc,tElapsed);
disp('Running the algorithm in its pure Matlab implementation')
disp('Parameters k: 10, distance: Normalized Euclidean, others: Default.');
tStart = tic;
owner = pcc(X, slabel, 10, 'seuclidean');
tElapsed = toc(tStart);
% Evaluating the classification accuracy.
acc = stmwevalk(label,slabel,owner);
fprintf('Classification accuracy: %0.4f - Execution Time: %0.4fs\n\n',acc,tElapsed);
disp('Notice that classification accuracy may vary between the two different implementations and among successive executions.');
disp('This is expected behavior due to the algorithm''s stochastic nature.');