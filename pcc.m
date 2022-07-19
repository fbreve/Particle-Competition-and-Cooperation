% Semi-Supervised Learning with Particle Competition and Cooperation
% by Fabricio Breve - 21/01/2019
%
% If you use this algorithm, please cite:
% Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gon�alves;
% Pedrycz, Witold; Liu, Jiming, "Particle Competition and Cooperation in
% Networks for Semi-Supervised Learning," Knowledge and Data Engineering,
% IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012.
% doi: 10.1109/TKDE.2011.119
%
% Usage: [owner, pot, owndeg, distnode] = pcc(X, slabel, options)
% INPUT:
% X         - Matrix where each line is a data item and each column is an
%             attribute
% slabel    - Vector where each element is the label of the corresponding
%             data item in X (use 1,2,3,... for labeled data items and 0
%             for unlabeled data items)
% OPTIONS:
% k         - Each node is connected to its k-neirest neighbors. Default:
%             size of the dataset multiplied by 0.05.
% disttype  - Use 'euclidean', 'seuclidean', etc. Default: 'euclidean'
% valpha    - Lower it to stop earlier, accuracy may be lower. Default:
%             2000.
% pgrd      - Check p_grd in [1]. Default: 0.5.
% deltav    - Check delta_v in [1]. Default: 0.1
% deltap    - Default: 1 (leave it on default to match equations in [1])
% dexp      - Default: 2 (leave it on default to match equations in [1])
% nclass    - Amount of classes on the problem. Default is the highest
%             label number in slabel.
% maxiter   - Maximum amount of iterations. Default is 500,000.
% mex       - Uses the mex version of the code (compiled binary) which is
%             ~10 times faster. Default: true. Set to false to use the 
%             Matlab only version.
% seed      - If seed is a nonnegative integer, it will be used to itialize
%             the random number generator, which is rng() or rand_s() 
%             in non-mex and mex versions, respectively. Use a seed if you
%             want reproducible results.
%
% OUTPUT:
% owner     - vector of classes assigned to each data item
% owndeg    - fuzzy output as in [2], each line is a data item, each column
%             pertinence to a class
% distnode  - matrix with the distance vectors of each particle, each
%             column is a particle and each line is a node
%
% [1] Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gon�alves;
% Pedrycz, Witold; Liu, Jiming, "Particle Competition and Cooperation in
% Networks for Semi-Supervised Learning," Knowledge and Data Engineering,
% IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012.
% doi: 10.1109/TKDE.2011.119
%
% [2] Breve, Fabricio Aparecido; ZHAO, Liang. 
% "Fuzzy community structure detection by particle competition and cooperation."
% Soft Computing (Berlin. Print). , v.17, p.659 - 673, 2013.

function [owner, pot, owndeg, distnode] = pcc(X, slabel, options)
    arguments
        X double
        slabel uint16
        options.k uint16 = size(X,1)*0.05
        options.disttype string = 'euclidean'
        options.valpha double = 2000
        options.pgrd double = 0.500 % probability of taking the greedy movement
        options.deltav double = 0.100 % controls node domination levels increase/decrease rate
        options.deltap double = 1.000 % controls particle potential increase/decrease rate
        options.dexp double = 2 % probabilities exponential
        options.nclass uint16 = max(slabel) % quantity of classes
        options.maxiter uint32 = 500000 % maximum amount of iterations
        options.mex logical = true % uses the mex version
        options.seed int32 = -1 % random seed
    end     

    % constants
    potmax = 1.000; % potencial m�ximo
    potmin = 0.000; % potencial m�nimo
    npart = sum(slabel~=0); % quantidade de part�culas
    qtnode = size(X,1);
    stopmax = round((qtnode/double(npart*options.k))*round(options.valpha*0.1)); % qtde de itera��es para verificar converg�ncia  
    % normalizar atributos se necess�rio
    if strcmp(options.disttype,'seuclidean')==1
        X = zscore(X);
        options.disttype='euclidean';
    end
    % encontrando k-vizinhos mais pr�ximos      
    KNN = uint32(knnsearch(X,X,'K',options.k+1,'NSMethod','kdtree','Distance',options.disttype));
    KNN = KNN(:,2:end); % eliminando o elemento como vizinho de si mesmo
    KNN(:,end+1:end+options.k) = 0;
    knns = repmat(options.k,qtnode,1); % vetor com a quantidade de vizinhos de cada n�       
    for i=1:qtnode
        KNN(sub2ind(size(KNN),KNN(i,1:options.k),(knns(KNN(i,1:options.k))+1)'))=i; % adicionando i como vizinho dos vizinhos de i (criando reciprocidade)
        knns(KNN(i,1:options.k))=knns(KNN(i,1:options.k))+1; % aumentando contador de vizinhos nos n�s que tiveram vizinhos adicionados
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
    pot = repmat(potmax/double(options.nclass),qtnode,options.nclass);
    % zerando potenciais dos n�s rotulados
    pot(partnode,:) = 0;
    % ajustando potencial da classe respectiva do n� rotulado para 1
    pot(sub2ind(size(pot),partnode,slabel(partnode))) = 1;
    % colocando cada n� em sua casa
    partpos = partnode;           
    % definindo grau de propriedade
    owndeg = repmat(realmin,qtnode,options.nclass);  % n�o podemos usar 0, porque n�s n�o visitados dariam divis�o por 0
    
    if options.mex==true    
        pccloop(options.maxiter, npart, options.nclass, stopmax, options.pgrd, ...
            options.dexp, options.deltav, options.deltap, potmin, partnode, ... 
            partclass, potpart, slabel, knns, distnode, KNN, pot, owndeg, options.seed);
    else
        % if a seed was provided in the options, use it.
        if options.seed>0
            rng(options.seed)
        end
        % vari�vel para guardar m�ximo potencial mais alto m�dio
        maxmmpot = 0;
        % counter of how much times the stop criterion checked positive
        stopcnt = 0;
        for i=1:options.maxiter
            % para cada part�cula
            rndtb = unifrnd(0,1,npart,1);  % probabilidade pdet
            roulettepick = unifrnd(0,1,npart,1);  % sorteio da roleta
            for j=1:npart
                ppj = partpos(j);
                if rndtb(j)<options.pgrd
                    % regra de probabilidade
                    prob = cumsum((1./(1+double(distnode(KNN(ppj,1:knns(ppj)),j))).^options.dexp)'.* pot(KNN(ppj,1:knns(ppj)),partclass(j))');
                    % descobrindo quem foi o n� sorteado
                    k = KNN(ppj,find(prob>=(roulettepick(j)*prob(end)),1,'first'));
                else
                    k = KNN(ppj,ceil(roulettepick(j)*double(knns(ppj))));
                    % contador de visita (para calcular grau de propriedade)
                    owndeg(k,partclass(j)) = owndeg(k,partclass(j)) + potpart(j);
                end           
                % se o n� n�o � pr�-rotulado
                if slabel(k)==0
                    % calculando novos potenciais para n�
                    deltapotpart = pot(k,:) - max(potmin,pot(k,:) - potpart(j)*(options.deltav/(double(options.nclass)-1)));
                    pot(k,:) = pot(k,:) - deltapotpart;
                    pot(k,partclass(j)) = pot(k,partclass(j)) + sum(deltapotpart);
                end
                % atribui novo potencial para part�cula
                potpart(j) = potpart(j) + (pot(k,partclass(j))-potpart(j))*options.deltap;
                          
                % se dist�ncia do n� alvo maior que dist�ncia do n� atual + 1
                if distnode(partpos(j),j)+1<distnode(k,j)
                    % atualizar dist�ncia do n� alvo
                    distnode(k,j) = distnode(partpos(j),j)+1;
                end
                
                % se n�o houve choque
                if pot(k,partclass(j))>=max(pot(k,:))
                    % muda para n� alvo
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
    end
    [~,owner] = max(pot,[],2);
    owndeg = owndeg ./ repmat(sum(owndeg,2),1,options.nclass);
end

