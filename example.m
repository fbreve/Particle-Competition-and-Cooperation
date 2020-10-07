% This is an example of Particle Competition and Cooperation (PCC) method 
% applied to the Wine Data Set from the UCI Machine Learning Repository
% Available at: https://archive.ics.uci.edu/ml/datasets/wine
%
% by Fabricio Breve - 12/03/2019
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
% Setting the k parameter (k-nearest neighbors)
k = 10;

disp('Running the algorithm in its pure Matlab implementation...')
disp('Parameters k: 10, distance: Normalized Euclidean, others: Default.');
tStart = tic;
owner = pcc(X, slabel, k, 'seuclidean', 200);
tElapsed = toc(tStart);
% Evaluating the classification accuracy.
acc = stmwevalk(label,slabel,owner);
fprintf('Classification accuracy: %0.4f - Execution Time: %0.4fs\n\n',acc,tElapsed);

% The mex implementation needs pccloop.c. It is pre-compiled for Win64. 
% If you have other operating system you need to compile it yourself using:  
% mex pccloop.c

disp('Running the algorithm MEX implementation, which is ~10 times faster than the pure Matlab implementation...');
disp('Parameters k: 10, distance: Normalized Euclidean, others: Default.');
tStart = tic;
owner = pccmex(X, slabel, k, 'seuclidean', 200);
tElapsed = toc(tStart);
% Evaluating the classification accuracy.
acc = stmwevalk(label,slabel,owner);
fprintf('Classification accuracy: %0.4f - Execution Time: %0.4fs\n\n',acc,tElapsed);

% Notice that classification accuracy may vary between the two different 
% implementations and among successive executions.
% This is expected behavior due to the algorithm''s stochastic nature.