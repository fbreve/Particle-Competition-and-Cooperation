% Semi-Supervised Territory Mark Walk Evaluation
% including Kappa coefficient
% Usage: [acc,kap] = stmwevalk(label,slabel,owner);
% label = real labels
% slabel = pre-labeled labels (0 = no label)
% owner = strwalk output
% wl = [0 - do not compute labeled items; 1 - compute labeled items]
function [acc,k] = stmwevalk(label,slabel,owner,wl)
    if (nargin < 4) || isempty(wl)
        wl = 0;
    end
    if wl==0
        acc = sum(label==owner & slabel==0)/sum(slabel==0);
    else
        acc = sum(label==owner)/size(label,1);
    end
    % calculando matriz de confusão
    c = zeros(max(label));
    for i=1:size(label,1)
        if (wl==1 || slabel(i)==0)
            c(label(i),owner(i)) = c(label(i),owner(i)) + 1;
        end
    end
    k = kappa(c);
end