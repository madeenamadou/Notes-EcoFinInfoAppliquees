%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   cournot.m:  A Matlab routine function that computes the value 
%   fval and Jacobian fjac of the function at an arbitrary point x
%   ----
%   Youssef de Madeen Amadou, Winter 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [fval,fjac] = cournot(q)
    c = [0.6; 0.8]; eta = 1.6; e = -1/eta;
    fval = sum(q)^e + e*sum(q)^(e-1)*q - diag(c)*q;
    fjac = e*sum(q)^(e-1)*ones(2,2) + e*sum(q)^(e-1)*eye(2) ...
    + (e-1)*e*sum(q)^(e-2)*q*[1 1] - diag(c);
end
