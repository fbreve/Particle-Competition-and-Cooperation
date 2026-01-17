% benchmark_pcc.m
% Benchmark PCC: pure MATLAB vs MEX, com 100 splits diferentes de pré-rotulados
% Baseado no exemplo original do Wine.
%
% Some results:
%
% Machine: Intel Core i9 14900K with 128GB of RAM
% Software: MATLAB R2025b
%
% With use_earlystop = true and N_RUNS = 100
%
% === Wine dataset (n = 178, d = 13) ===
% Running pure MATLAB PCC (multiple runs)...
% Pure MATLAB PCC: mean time = 0.3307s (std = 0.0870s), mean acc = 0.9355
% Running MEX PCC (multiple runs)...
% MEX PCC:         mean time = 0.0136s (std = 0.0031s), mean acc = 0.9359
% 
% Speedup (MATLAB / MEX): 24.4x
% Loading the Digits dataset...
% 
% === Digits dataset (n = 1797, d = 61) ===
% Running pure MATLAB PCC (multiple runs)...
% Pure MATLAB PCC: mean time = 8.7377s (std = 2.2153s), mean acc = 0.9550
% Running MEX PCC (multiple runs)...
% MEX PCC:         mean time = 0.3566s (std = 0.0903s), mean acc = 0.9550
% 
% Speedup (MATLAB / MEX): 24.5x
%
% With use_earlystop = false and N_RUNS = 10
%
% === Wine dataset (n = 178, d = 13) ===
% Running pure MATLAB PCC (multiple runs)...
% Pure MATLAB PCC: mean time = 30.6969s (std = 0.1019s), mean acc = 0.9187
% Running MEX PCC (multiple runs)...
% MEX PCC:         mean time = 0.9722s (std = 0.0050s), mean acc = 0.9281
% 
% Speedup (MATLAB / MEX): 31.6x
% Loading the Digits dataset...
% 
% === Digits dataset (n = 1797, d = 61) ===
% Running pure MATLAB PCC (multiple runs)...
% Pure MATLAB PCC: mean time = 284.9530s (std = 1.9226s), mean acc = 0.9587
% Running MEX PCC (multiple runs)...
% MEX PCC:         mean time = 10.8661s (std = 0.0606s), mean acc = 0.9585
% 
% Speedup (MATLAB / MEX): 26.2x

clear; clc;

N_RUNS       = 10;
LABELED_FRAC = 0.1;
k            = 10;
use_earlystop  = false;   % <--- flag
valpha         = 2000;   % <--- equivalente ao Python

%% Wine dataset
disp('Loading the Wine dataset...');
load wine.data
X     = wine(:,2:end);
label = wine(:,1);

fprintf('\n=== Wine dataset (n = %d, d = %d) ===\n', size(X,1), size(X,2));

%% Pure MATLAB implementation
disp('Running pure MATLAB PCC (multiple runs)...');

t_list_mat   = zeros(N_RUNS,1);
acc_list_mat = zeros(N_RUNS,1);

for r = 1:N_RUNS
    rng(r);
    slabel = slabelgen(label, LABELED_FRAC);

    tStart = tic;
    owner = pcc(X, slabel, ...
                k=10, ...
                disttype='seuclidean', ...
                mex=false, ...
                earlystop=use_earlystop, ...
                valpha=valpha);
    tElapsed = toc(tStart);
    t_list_mat(r) = tElapsed;

    acc_list_mat(r) = stmwevalk(label, slabel, owner);
end

t_mean_mat   = mean(t_list_mat);
t_std_mat    = std(t_list_mat);
acc_mean_mat = mean(acc_list_mat);

fprintf('Pure MATLAB PCC: mean time = %.4fs (std = %.4fs), mean acc = %.4f\n', ...
        t_mean_mat, t_std_mat, acc_mean_mat);

%% MEX implementation
disp('Running MEX PCC (multiple runs)...');

t_list_mex   = zeros(N_RUNS,1);
acc_list_mex = zeros(N_RUNS,1);

for r = 1:N_RUNS
    rng(r);
    slabel = slabelgen(label, LABELED_FRAC);

    tStart = tic;
    owner = pcc(X, slabel, ...
                k=10, ...
                disttype='seuclidean', ...
                mex=true, ...
                earlystop=use_earlystop, ...
                valpha=valpha);
    tElapsed = toc(tStart);
    t_list_mex(r) = tElapsed;

    acc_list_mex(r) = stmwevalk(label, slabel, owner);
end

t_mean_mex   = mean(t_list_mex);
t_std_mex    = std(t_list_mex);
acc_mean_mex = mean(acc_list_mex);

fprintf('MEX PCC:         mean time = %.4fs (std = %.4fs), mean acc = %.4f\n', ...
        t_mean_mex, t_std_mex, acc_mean_mex);

%% Speedup
fprintf('\nSpeedup (MATLAB / MEX): %.1fx\n', t_mean_mat / t_mean_mex);

%% Digits dataset
disp('Loading the Digits dataset...');
% X: [1797 x 64], label: [1797 x 1] com dígitos 0..9 (já convertidos para 1..10)
load digits.mat   
% 3 colunas são zeradas, vamos encontrá-las e removê-las para evitar
% warnings
mask = ~all(X == 0, 1); 
X = X(:, mask);

fprintf('\n=== Digits dataset (n = %d, d = %d) ===\n', size(X,1), size(X,2));

% Pure MATLAB implementation
disp('Running pure MATLAB PCC (multiple runs)...');
t_list_mat   = zeros(N_RUNS,1);
acc_list_mat = zeros(N_RUNS,1);
for r = 1:N_RUNS
    rng(r);
    slabel = slabelgen(label, LABELED_FRAC);
    tStart = tic;
    owner = pcc(X, slabel, ...
                k=10, ...
                disttype='seuclidean', ...
                mex=false, ...
                earlystop=use_earlystop, ...
                valpha=valpha);
    tElapsed = toc(tStart);
    t_list_mat(r)   = tElapsed;
    acc_list_mat(r) = stmwevalk(label, slabel, owner);
end
t_mean_mat   = mean(t_list_mat);
t_std_mat    = std(t_list_mat);
acc_mean_mat = mean(acc_list_mat);
fprintf('Pure MATLAB PCC: mean time = %.4fs (std = %.4fs), mean acc = %.4f\n', ...
        t_mean_mat, t_std_mat, acc_mean_mat);

% MEX implementation
disp('Running MEX PCC (multiple runs)...');
t_list_mex   = zeros(N_RUNS,1);
acc_list_mex = zeros(N_RUNS,1);
for r = 1:N_RUNS
    rng(r);
    slabel = slabelgen(label, LABELED_FRAC);
    tStart = tic;
    owner = pcc(X, slabel, ...
                k=10, ...
                disttype='seuclidean', ...
                mex=true, ...
                earlystop=use_earlystop, ...
                valpha=valpha);
    tElapsed = toc(tStart);
    t_list_mex(r)   = tElapsed;
    acc_list_mex(r) = stmwevalk(label, slabel, owner);
end
t_mean_mex   = mean(t_list_mex);
t_std_mex    = std(t_list_mex);
acc_mean_mex = mean(acc_list_mex);
fprintf('MEX PCC:         mean time = %.4fs (std = %.4fs), mean acc = %.4f\n', ...
        t_mean_mex, t_std_mex, acc_mean_mex);

fprintf('\nSpeedup (MATLAB / MEX): %.1fx\n', t_mean_mat / t_mean_mex);
