% benchmark_pcc.m
% Benchmark PCC: pure MATLAB vs MEX, com 100 splits diferentes de pré-rotulados
% Baseado no exemplo original do Wine.
%
% Some results:
%
% Machine: Intel Core i9 14900K with 128GB of RAM
% Software: MATLAB R2025b
%
% With early_stop = true
%
% Pure MATLAB PCC: mean time = 1.1467s (std = 0.6377s), mean acc = 0.9355
% MEX PCC:         mean time = 0.0398s (std = 0.0091s), mean acc = 0.9346
% Speedup (MATLAB / MEX): 28.8x
%
% With early_stop = false
%
% Pure MATLAB PCC: mean time = 34.4782s (std = 1.2620s), mean acc = 0.9376
% MEX PCC:         mean time = 0.9996s (std = 0.0091s), mean acc = 0.9357
% Speedup (MATLAB / MEX): 34.5x

clear; clc;

N_RUNS       = 100;
LABELED_FRAC = 0.1;
k            = 10;
use_earlystop  = true;   % <--- flag
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
