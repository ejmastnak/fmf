function [V, A]=eig_householder(A, eps)
% Diagonalizes the inputted matrix A using Householder transformations
% param A: nxn symmetric matrix
% param eps: scalar tolerance for stopping iteration, compared to off(A)
% return A: near-diagonal nxn matrix whose diagonal elements are A's
% eigenvalues. Formally would be named D, but why create a new variable and
% waste memory when you can just use A?
% return V: nxn matrix whose column vectors are A's eigenvectors

n = length(A);

V = eye(n);

k = 1;
while offset(A) > eps
    for j=1:(n-1)
        P = householder_eig(A, j, j);
        A = P*A*P;  % note that P = P^T
        V = V*P;
        k = k+1;
    end
end
    
end