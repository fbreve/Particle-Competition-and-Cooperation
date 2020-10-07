% Semi-Supervised Learning with Particle Competition and Cooperation
% by Fabricio Breve - 21/01/2019
%
% This MEX version is ~10 times faster than the regular version
%
% If you use this algorithm, please cite:
% Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gon�alves; Pedrycz, Witold; Liu, Jiming, 
% "Particle Competition and Cooperation in Networks for Semi-Supervised Learning," 
% Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012
% doi: 10.1109/TKDE.2011.119
%
% Usage: [owner, pot, owndeg] = pccmex(X, slabel, k, disttype, valpha, pgrd, deltav, deltap, dexp, nclass, maxiter)
% INPUT:
% X         - Matrix where each line is a data item and each column is an attribute
% slabel    - vector where each element is the label of the corresponding
%             data item in X (use 1,2,3,... for labeled data items and 0
%             for unlabeled data items)
% k         - each node is connected to its k-neirest neighbors
% disttype  - use 'euclidean', 'seuclidean', etc.
% valpha    - Default: 2000 (lower it to stop earlier, accuracy may be lower)
% pgrd      - check p_grd in [1]
% deltav    - check delta_v in [1]
% deltap    - Default: 1 (leave it on default to match equations in [1])
% dexp      - Default: 2 (leave it on default to match equations in [1])
% nclass    - amount of classes on the problem
% maxiter   - maximum amount of iterations
% OUTPUT:
% owner     - vector of classes assigned to each data item
% owndeg    - fuzzy output as in [2], each line is a data item, each column pertinence to a class
%
% [1] Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gon�alves; Pedrycz, Witold; Liu, Jiming, 
% "Particle Competition and Cooperation in Networks for Semi-Supervised Learning," 
% Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012
% doi: 10.1109/TKDE.2011.119
%
% [2] Breve, Fabricio Aparecido; ZHAO, Liang. 
% "Fuzzy community structure detection by particle competition and cooperation."
% Soft Computing (Berlin. Print). , v.17, p.659 - 673, 2013.
function [owner, pot, owndeg] = pccmex(X, slabel, k, disttype, valpha, pgrd, deltav, deltap, dexp, nclass, maxiter)
    if (nargin < 11) || isempty(maxiter)
        maxiter = 500000; % n�mero de itera��es
    end
    if (nargin < 10) || isempty(nclass)
        nclass = max(slabel); % quantidade de classes
    end
    if (nargin < 9) || isempty(dexp)
        dexp = 2; % exponencial de probabilidade
    end
    if (nargin < 8) || isempty(deltap)
        deltap = 1.000; % controle de velocidade de aumento/decremento do potencial da part�cula
    end
    if (nargin < 7) || isempty(deltav)
        deltav = 0.100; % controle de velocidade de aumento/decremento do potencial do v�rtice
    end
    if (nargin < 6) || isempty(pgrd)
        pgrd = 0.500; % probabilidade de n�o explorar
    end
    if (nargin < 5) || isempty(valpha)
        valpha = 2000;
    end    
    if (nargin < 4) || isempty(disttype)
        disttype = 'euclidean'; % dist�ncia euclidiana n�o normalizada
    end    
    qtnode = size(X,1); % quantidade de n�s
    if (nargin < 3) || isempty(k)
        k = round(qtnode*0.05); % quantidade de vizinhos mais pr�ximos
    end
    % tratamento da entrada
    slabel = uint16(slabel);
    k = uint16(k);
    % constantes
    potmax = 1.000; % potencial m�ximo
    potmin = 0.000; % potencial m�nimo
    npart = sum(slabel~=0); % quantidade de part�culas
    stopmax = round((qtnode/(npart*k))*round(valpha*0.1)); % qtde de itera��es para verificar converg�ncia    
    % normalizar atributos se necess�rio
    if strcmp(disttype,'seuclidean')==1
        X = zscore(X);
        disttype='euclidean';
    end
    % encontrando k-vizinhos mais pr�ximos      
    KNN = uint32(knnsearch(X,X,'K',k+1,'NSMethod','kdtree','Distance',disttype));
    KNN = KNN(:,2:end); % eliminando o elemento como vizinho de si mesmo
    KNN(:,end+1:end+k) = 0;
    knns = repmat(k,qtnode,1); % vetor com a quantidade de vizinhos de cada n�    
    for i=1:qtnode
        %KNNR(sub2ind(size(KNNR),KNN(i,:),(knns(KNN(i,:))+1)'))=i; % adicionando i como vizinho dos vizinhos de i (criando reciprocidade)
        KNN(sub2ind(size(KNN),KNN(i,1:k),(knns(KNN(i,1:k))+1)'))=i; % adicionando i como vizinho dos vizinhos de i (criando reciprocidade)
        knns(KNN(i,1:k))=knns(KNN(i,1:k))+1; % aumentando contador de vizinhos nos n�s que tiveram vizinhos adicionados
        if max(knns)==size(KNN,2) % se algum n� atingiu o limite de colunas da matriz de vizinhan�a rec�proca teremos de aument�-la
            KNN(:,max(knns)+1:round(max(knns)*1.1)+1) = zeros(qtnode,round(max(knns)*0.1)+1,'uint32');  % portanto vamos aumenta-la em 10% + 1 (para garantir no caso do tamanho ser menor que 10)            
        end
    end
    % removendo duplicatas    
    for i=1:qtnode
        knnrow = unique(KNN(i,:),'stable'); % remove as duplicatas
        knns(i) = size(knnrow,2)-1; % atualiza quantidade de vizinhos (e descarta o zero no final)
        KNN(i,1:knns(i)) = knnrow(1:end-1); % copia para matriz KNN
        %KNN(i,knns(i)+1:end) = 0; % preenche espa�os n�o usados por zero,
        %apenas para debug pois na pr�tica n�o faz diferen�a visto que knns
        %j� dir� quais s�o os vizinhos v�lidos da lista
    end    
    KNN = KNN(:,1:max(knns)); % eliminando colunas que n�o tem vizinhos v�lidos
    % definindo classe de cada part�cula
    partclass = slabel(slabel~=0);
    % definindo n� casa da part�cula
    partnode = uint32(find(slabel));
    % definindo potencial da part�cula em 1
    potpart = ones(potmax,npart);
    % ajustando todas as dist�ncias na m�xima poss�vel
    distnode = repmat(min(intmax('uint8'),uint8(qtnode-1)),qtnode,npart);
    % ajustando para zero a dist�ncia de cada part�cula para seu
    % respectivo n� casa
    distnode(sub2ind(size(distnode),partnode',1:npart)) = 0;
    % inicializando tabela de potenciais com tudo igual
    pot = repmat(potmax/nclass,qtnode,nclass);
    % zerando potenciais dos n�s rotulados
    pot(partnode,:) = 0;
    % ajustando potencial da classe respectiva do n� rotulado para 1
    pot(sub2ind(size(pot),partnode,slabel(partnode))) = 1;
    % colocando cada n� em sua casa
    %partpos = partnode;           
    owndeg = repmat(realmin,qtnode,nclass);    
    pccloop(maxiter, npart, nclass, stopmax, pgrd, dexp, deltav, deltap, potmin, partnode, partclass, potpart, slabel, knns, distnode, KNN, pot, owndeg);
    [~,owner] = max(pot,[],2);
    owndeg = owndeg ./ repmat(sum(owndeg,2),1,nclass);
end

