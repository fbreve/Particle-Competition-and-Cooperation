% Slabel Generator receives a vector where each element is a class number 
% (>0) and it outputs the same vector changing some labels to zero 
% (representing the unlabeled elements a classifier will estimate)
% It guarantees at least one labeled node per class
%
% Usage: slabel = slabelgen(label,amount);
% label = vector where each element is the label of an element represented
% by positive integers (1, 2, 3, ...)
% amount = [0 1] percentage of pre-labeled samples. Example: use 0.1 to 
% keep 10% of the labels unchanged and change the other 90% to zero.

function slabel = slabelgen(label,amount)
    qtnode = size(label,1);    
    slabel = zeros(qtnode,1);
    plabc = round(qtnode*amount);
    qtclass = max(label);
    for i=1:qtclass
        while 1 
            r = random('unid',qtnode);    
            if label(r)==i 
                break;
            end
        end
        slabel(r)=i;
        plabc = plabc - 1;
    end
    while plabc>0
        r = random('unid',qtnode);
        if slabel(r)==0
            slabel(r) = label(r);
            plabc = plabc - 1;
        end
    end
end
