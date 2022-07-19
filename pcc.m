% Semi-Supervised Learning with Particle Competition and Cooperation
% by Fabricio Breve - 21/01/2019
%
% If you use this algorithm, please cite:
% Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming, 
% "Particle Competition and Cooperation in Networks for Semi-Supervised Learning," 
% Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012
% doi: 10.1109/TKDE.2011.119
%
% Usage: [owner, pot, owndeg, distnode] = pcc(X, slabel, k, disttype, valpha, pgrd, deltav, deltap, dexp, nclass, maxiter)
% INPUT:
% X         - Matrix where each line is a data item and each column is an attribute
% slabel    - vector where each element is the label of the corresponding
%             data item in X (use 1,2,3,... for labeled data items and 0
%             for unlabeled data items)
% k         - each node is connected to its k-neirest neighbors
% disttype  - use 'euclidean', 'seuclidean', etc.
% valpha    - lower it to stop earlier, accuracy may be lower
% pgrd      - check p_grd in [1]
% deltav    - check delta_v in [1]
% deltap    - Default: 1 (leave it on default to match equations in [1])
% dexp      - Default: 2 (leave it on default to match equations in [1])
% nclass    - amount of classes on the problem
% maxiter   - maximum amount of iterations
% OUTPUT:
% owner     - vector of classes assigned to each data item
% owndeg    - fuzzy output as in [2], each line is a data item, each column pertinence to a class
% distnode  - matrix with the distance vectors of each particle, each
%             column is a particle and each line is a node
%
% [1] Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming, 
% "Particle Competition and Cooperation in Networks for Semi-Supervised Learning," 
% Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012
% doi: 10.1109/TKDE.2011.119
%
% [2] Breve, Fabricio Aparecido; ZHAO, Liang. 
% "Fuzzy community structure detection by particle competition and cooperation."
% Soft Computing (Berlin. Print). , v.17, p.659 - 673, 2013.
function [owner, pot, owndeg, distnode] = pcc(X, slabel, k, disttype, valpha, pgrd, deltav, deltap, dexp, nclass, maxiter)
    if (nargin < 11) || isempty(maxiter)
        maxiter = 500000; % número de iterações
    end
    if (nargin < 10) || isempty(nclass)
        nclass = max(slabel); % quantidade de classes
    end
    if (nargin < 9) || isempty(dexp)
        dexp = 2; % exponencial de probabilidade
    end
    if (nargin < 8) || isempty(deltap)
        deltap = 1.000; % controle de velocidade de aumento/decremento do potencial da partícula
    end
    if (nargin < 7) || isempty(deltav)
        deltav = 0.100; % controle de velocidade de aumento/decremento do potencial do vértice
    end
    if (nargin < 6) || isempty(pgrd)
        pgrd = 0.500; % probabilidade de não explorar
    end
    if (nargin < 5) || isempty(valpha)
        valpha = 2000;
    end    
    if (nargin < 4) || isempty(disttype)
        disttype = 'euclidean'; % distância euclidiana não normalizada
    end    
    qtnode = size(X,1); % quantidade de nós
    if (nargin < 3) || isempty(k)
        k = round(qtnode*0.05); % quantidade de vizinhos mais próximos
    end    
    % constantes
    potmax = 1.000; % potencial máximo
    potmin = 0.000; % potencial mínimo
    npart = sum(slabel~=0); % quantidade de partículas
    stopmax = round((qtnode/(npart*k))*round(valpha*0.1)); % qtde de iterações para verificar convergência    
    % normalizar atributos se necessário
    if strcmp(disttype,'seuclidean')==1
        X = zscore(X);
        disttype='euclidean';
    end
    % encontrando k-vizinhos mais próximos      
    KNN = knnsearch(X,X,'K',k+1,'NSMethod','kdtree','Distance',disttype);
    %KNN = knnsearch(X,X,'K',k+1,'Distance',disttype);
    KNN = KNN(:,2:end); % eliminando o elemento como vizinho de si mesmo
    KNNR = zeros(qtnode,k); % criando matriz para vizinhança recíproca, inicialmente com tamanho k
    knns = zeros(qtnode,1); % vetor com a quantidade de vizinhos recíprocos de cada nó
    for i=1:qtnode
        KNNR(sub2ind(size(KNNR),KNN(i,:),(knns(KNN(i,:))+1)'))=i; % adicionando i como vizinho dos vizinhos de i (criando reciprocidade)
        knns(KNN(i,:))=knns(KNN(i,:))+1; % aumentando contador de vizinhos nos nós que tiveram vizinhos adicionados
        if max(knns)==size(KNNR,2) % se algum nó atingiu o limite de colunas da matriz de vizinhança recíproca teremos de aumentá-la
            KNNR(:,max(knns)+1:max(knns)*2) = zeros(qtnode,max(knns));  % portanto vamos dobrá-la
        end
    end
    KNN = [KNN KNNR];
    clear KNNR;
    % removendo duplicatas    
    for i=1:qtnode
        knnrow = unique(KNN(i,:),'stable'); % remove as duplicatas
        knns(i) = size(knnrow,2)-1; % atualiza quantidade de vizinhos (e descarta o zero no final)
        KNN(i,1:knns(i)) = knnrow(1:end-1); % copia para matriz KNN
        %KNN(i,knns(i)+1:end) = 0; % preenche espaços não usados por zero,
        %apenas para debug pois na prática não faz diferença visto que knns
        %já dirá quais são os vizinhos válidos da lista
    end    
    KNN = KNN(:,1:max(knns)); % eliminando colunas que não tem vizinhos válidos
    % definindo classe de cada partícula
    partclass = slabel(slabel~=0);
    % definindo nó casa da partícula
    partnode = find(slabel);
    % definindo potencial da partícula em 1
    potpart = repmat(potmax,npart,1);       
    % ajustando todas as distâncias na máxima possível
    distnode = repmat(qtnode-1,qtnode,npart);
    % ajustando para zero a distância de cada partícula para seu
    % respectivo nó casa
    distnode(sub2ind(size(distnode),partnode',1:npart)) = 0;
    % inicializando tabela de potenciais com tudo igual
    pot = repmat(potmax/nclass,qtnode,nclass);
    % zerando potenciais dos nós rotulados
    pot(partnode,:) = 0;
    % ajustando potencial da classe respectiva do nó rotulado para 1
    pot(sub2ind(size(pot),partnode,slabel(partnode))) = 1;
    % colocando cada nó em sua casa
    partpos = partnode;           
    % definindo grau de propriedade
    owndeg = repmat(realmin,qtnode,nclass);  % não podemos usar 0, porque nós não visitados dariam divisão por 0
    % variável para guardar máximo potencial mais alto médio
    maxmmpot = 0;
    % counter of how much times the stop criterion checked positive
    stopcnt = 0;
    for i=1:maxiter
        % para cada partícula
        rndtb = unifrnd(0,1,npart,1);  % probabilidade pdet
        roulettepick = unifrnd(0,1,npart,1);  % sorteio da roleta
        for j=1:npart
            ppj = partpos(j);
            if rndtb(j)<pgrd
                % regra de probabilidade                
                prob = cumsum((1./(1+distnode(KNN(ppj,1:knns(ppj)),j)).^dexp)'.* pot(KNN(ppj,1:knns(ppj)),partclass(j))');               
                % descobrindo quem foi o nó sorteado
                k = KNN(ppj,find(prob>=(roulettepick(j)*prob(end)),1,'first'));
            else
                k = KNN(ppj,ceil(roulettepick(j)*knns(ppj)));
                % contador de visita (para calcular grau de propriedade)
                owndeg(k,partclass(j)) = owndeg(k,partclass(j)) + potpart(j);
            end           
            % se o nó não é pré-rotulado
            if slabel(k)==0
                % calculando novos potenciais para nó
                deltapotpart = pot(k,:) - max(potmin,pot(k,:) - potpart(j)*(deltav/(nclass-1)));
                pot(k,:) = pot(k,:) - deltapotpart;
                pot(k,partclass(j)) = pot(k,partclass(j)) + sum(deltapotpart);
            end
            % atribui novo potencial para partícula
            potpart(j) = potpart(j) + (pot(k,partclass(j))-potpart(j))*deltap;
                      
            % se distância do nó alvo maior que distância do nó atual + 1
            if distnode(partpos(j),j)+1<distnode(k,j)
                % atualizar distância do nó alvo
                distnode(k,j) = distnode(partpos(j),j)+1;
            end
            
            % se não houve choque
            if pot(k,partclass(j))>=max(pot(k,:))
                % muda para nó alvo
                partpos(j) = k;
            end
        end
        if mod(i,10)==0
            mmpot = mean(max(pot,[],2));
            %disp(sprintf('Iter: %5.0f  Meanpot: %0.4f',i,mmpot))
            if mmpot>maxmmpot
                maxmmpot = mmpot;
                stopcnt = 0;
            else    
                stopcnt = stopcnt + 1;
                if stopcnt > stopmax
                    break;
                end
            end
        end
    end
    [~,owner] = max(pot,[],2);
    owndeg = owndeg ./ repmat(sum(owndeg,2),1,nclass);
end

