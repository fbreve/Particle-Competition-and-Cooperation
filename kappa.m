% ====================================================================
% KAPPA - [k,var] = kappa(C)
%
%  var is the variance of the coefficient
%  Avaliates the performance of the classification 
%  by the kappa coeficient.  C is the confusion matrix of the
%  classification.
%
% Ronaldo L. Alonso
% ====================================================================

function [k,var]=kappa(C)
  [rc,cc] = size(C);
  
  if (rc ~= cc)
      error('Matrix must be square.');
  end    
  cols = sumcol(C);
  lins = sumcol(C');
  N = sum(cols);
  theta1 = sum(diag(C))/N;
  theta2 = dot(lins,cols)/(N^2);
  theta3 = 0;

  for i=1:rc
    theta3 = theta3 + C(i,i)*(lins(i)+cols(i));
  end
  theta3 = theta3/(N^2);
  theta4 = 0;
  
  for j = 1:rc
    for i=1:rc
      theta4 = theta4 + C(i,j)*(lins(j) + cols(i))^2; 
    end
  end
  theta4 = theta4/(N^3);
  var = (theta1*(1-theta1)/(1-theta2)^2 + 2*(1-theta1)*(2*theta1*theta2 -theta3) /(1- theta2)^3 +(1-theta1)^2*(theta4 -4*theta2^2)/ (1-theta2)^4);
  var = var/N;
  k = (theta1-theta2)/(1-theta2);
return