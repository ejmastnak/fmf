function [V, D]=eig_qr(A, eps)
% Diagonalizes the inputted matrix A using QR iteration
% param A: nxn symmetric matrix
% param eps: scalar tolerance for stopping iteration, compared to off(A)
% return D: near-diagonal nxn matrix whose diagonal elements are A's
% eigenvalues. 
% return V: nxn orthogonal transformation matrix whose column vectors are
% A's eigenvalues and for which D = V'AV

n = length(A);
D = A;
V = eye(n);
while offset(D) > eps
    [Q, R] = qr_givens(D);
    % [Q, R] = qr(D);  % built-in QR
    D = R*Q;
    V = V*Q;
end
    
end