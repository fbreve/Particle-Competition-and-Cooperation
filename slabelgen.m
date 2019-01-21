% Gerador de slabel (n�s rotulados = classe, n�s n�o rotulados = 0)
% Garante pelo menos 1 n� rotulado por classe
% slabel = slabelgen(label,amount);
% label = label list
% amount = [0 1] percentage of pre-labeled samples
% Uso: slabel = slabelgen(label,amount)
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
