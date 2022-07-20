% Semi-Supervised Learning with Particle Competition and Cooperation
% by Fabricio Breve - 21/01/2019
%
% If you use this algorithm, please cite:
% Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves;
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
% useseed   - Set to true if you want to use a seed to allow reproducible
%             results. Default: false. Remember to set the seed with the
%             seed option.
% seed      - Seed to itialize the random number generator, which is rng()
%             or rand_s() in non-mex and mex versions, respectively. 
%             Default: 0. Remember to set the option useseed to true
%             if you want the provided seed to be used. 
% Xgraph    - Set to true if X is a pre-built graph instead of a feature
%             matrix.
%
% OUTPUT:
% owner     - vector of classes assigned to each data item
% owndeg    - fuzzy output as in [2], each line is a data item, each column
%             pertinence to a class
% distnode  - matrix with the distance vectors of each particle, each
%             column is a particle and each line is a node
%
% [1] Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves;
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
        X 
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
        options.useseed logical = false % do not set seed
        options.seed int32 = 0 % random seed
        options.Xgraph = false % X is a feature matrix
    end     

    % if the input X is a feature matrix (default)
    if options.Xgraph == false
        % call the function to build the graph
        G = pcc_buildgraph(double(X),k=options.k, disttype=options.disttype);
        qtnode = size(X,1); % amount of nodes
        k = options.k;
    % if the input X is a pre-built graph
    else        
        G = X;
        qtnode = size(G.KNN,1); % amount of nodes
        k = G.k;
    end

    % constants
    potmax = 1.000; % maximum dominance level
    potmin = 0.000; % minimum dominance level
    npart = sum(slabel~=0); % amount of particles
    
    % amount of iterations to check for convergence
    stopmax = round((qtnode/double(npart*k))*round(options.valpha*0.1));

    % definining the class of each particle
    partclass = slabel(slabel~=0);
    % defining the home node of each particle
    partnode = uint32(find(slabel));
    % definindo the strength of each particle to 1
    potpart = ones(potmax,npart);       
    % adjusting all distance in particles distance tables to the maximum
    distnode = repmat(min(intmax('uint8'),uint8(qtnode-1)),qtnode,npart);
    % adjusting to zero the distance of each particle to its home node
    distnode(sub2ind(size(distnode),partnode',1:npart)) = 0;
    % initializing all the dominance vectors with the same levels (1/amount
    % of classes)
    pot = repmat(potmax/double(options.nclass),qtnode,options.nclass);
    % zero-ing the dominance vectors of labeled nodes
    pot(partnode,:) = 0;
    % ajusting the dominance vector of the nodes' classes to 1
    pot(sub2ind(size(pot),partnode,slabel(partnode))) = 1;
    % putting each particle in its home node
    partpos = partnode;           
    % initializing the accumlated dominance vectors
    % we can't use 0, otherwise non-visited nodes would trigger a division
    % by zero.
    owndeg = repmat(realmin,qtnode,options.nclass);  
    
    % if the mex version was chosen
    if options.mex==true    
        pccloop(options.maxiter, npart, options.nclass, stopmax, options.pgrd, ...
            options.dexp, options.deltav, options.deltap, potmin, partnode, ... 
            partclass, potpart, slabel, G.knns, distnode, G.KNN, ...
            pot, owndeg, options.useseed, options.seed);
    % if the non-mex version was chosen
    else
        % if a seed was provided in the options, use it.
        if options.useseed==true
            rng(options.seed)
        end
        % variável para guardar máximo potencial mais alto médio
        maxmmpot = 0;
        % counter of how much times the stop criterion checked positive
        stopcnt = 0;
        for i=1:options.maxiter
            % para cada partícula
            rndtb = unifrnd(0,1,npart,1);  % probabilidade pdet
            roulettepick = unifrnd(0,1,npart,1);  % sorteio da roleta
            for j=1:npart
                ppj = partpos(j);
                if rndtb(j)<options.pgrd
                    % regra de probabilidade
                    prob = cumsum((1./(1+double(distnode(G.KNN(ppj,1:G.knns(ppj)),j))).^options.dexp)'.* pot(G.KNN(ppj,1:G.knns(ppj)),partclass(j))');
                    % descobrindo quem foi o nó sorteado
                    picked = G.KNN(ppj,find(prob>=(roulettepick(j)*prob(end)),1,'first'));
                else
                    picked = G.KNN(ppj,ceil(roulettepick(j)*double(G.knns(ppj))));
                    % contador de visita (para calcular grau de propriedade)
                    owndeg(picked,partclass(j)) = owndeg(picked,partclass(j)) + potpart(j);
                end           
                % se o nó não é pré-rotulado
                if slabel(picked)==0
                    % calculando novos potenciais para nó
                    deltapotpart = pot(picked,:) - max(potmin,pot(picked,:) - potpart(j)*(options.deltav/(double(options.nclass)-1)));
                    pot(picked,:) = pot(picked,:) - deltapotpart;
                    pot(picked,partclass(j)) = pot(picked,partclass(j)) + sum(deltapotpart);
                end
                % atribui novo potencial para partícula
                potpart(j) = potpart(j) + (pot(picked,partclass(j))-potpart(j))*options.deltap;
                          
                % se distância do nó alvo maior que distância do nó atual + 1
                if distnode(partpos(j),j)+1<distnode(picked,j)
                    % atualizar distância do nó alvo
                    distnode(picked,j) = distnode(partpos(j),j)+1;
                end
                
                % se não houve choque
                if pot(picked,partclass(j))>=max(pot(picked,:))
                    % muda para nó alvo
                    partpos(j) = picked;
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

