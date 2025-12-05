% Usage: pcc_graph = pcc_buildgraph(X, options)
% INPUT:
% X         - Matrix where each line is a data item and each column is an
%             attribute
% OPTIONS:
% k         - Each node is connected to its k-neirest neighbors. Default:
%             size of the dataset multiplied by 0.05.
% disttype  - Use 'euclidean', 'seuclidean', etc. Default: 'euclidean'.
%             See the MATLAB knnsearch funcion help for all the options.
%
% OUTPUT:
% pcc_graph - structure containing KNN and knns
%             KNN is a matrix of neighbors, each line is a node, each
%             column is a neighbor.
%             knns is a vector where each element is the amount of
%             neighbors of a node.

function pcc_graph = pcc_buildgraph(X, options)
    arguments
        X double
        options.k uint16 = size(X,1)*0.05
        options.disttype string = 'euclidean'
    end     

    qtnode = size(X,1);
    % find the k-nearest neighbors
    KNN = uint32(knnsearch(X,X,'K',options.k+1,'Distance',options.disttype));
    % eliminate the self-loops
    KNN = KNN(:,2:end); 
    % make room for reciprocal connections
    KNN(:,end+1:end+options.k) = 0; 
    % itialize vector holding the amount of neighbors of each node.
    % before the reciprocal connections, all nodes have k neighbors.
    knns = repmat(options.k,qtnode,1);
    %
    for i=1:qtnode
        % adding i as neighbor of its neighbors (creating reciprocity)
        KNN(sub2ind(size(KNN),KNN(i,1:options.k),(knns(KNN(i,1:options.k))+1)'))=i; 
        % increasing neighbors counter for nodes that had neighbors added
        knns(KNN(i,1:options.k))=knns(KNN(i,1:options.k))+1; 
        % if any node has as many neighbors as the matrix width
        if max(knns)==size(KNN,2)
            % increase the matrix width by 10% + 1
            KNN(:,max(knns)+1:round(max(knns)*1.1)+1) = zeros(qtnode,round(max(knns)*0.1)+1,'uint32');
        end
    end
    % for all nodes
    for i=1:qtnode
        % remove duplicate neighbors
        knnrow = unique(KNN(i,:),'stable'); 
        % update the neighbors amount (and discard the zero at the end)
        knns(i) = size(knnrow,2)-1; 
        % copy the results to the KNN matrix
        KNN(i,1:knns(i)) = knnrow(1:end-1);
        %KNN(i,knns(i)+1:end) = 0; % fill non-used spaces with zero,
        % only for debugging since knns will tell which positions are valid
        % in the list        
    end    
    % eliminating columns without valid neighbors
    KNN = KNN(:,1:max(knns)); 
    % save the matrix of neighbors and the vector of the amount of
    % neighbors in the structure to be returned.
    pcc_graph.KNN = KNN;
    pcc_graph.knns = knns;
    pcc_graph.k = options.k;
end

