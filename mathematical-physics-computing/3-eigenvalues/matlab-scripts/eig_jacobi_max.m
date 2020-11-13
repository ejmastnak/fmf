function [V, D]=eig_jacobi_max(A, eps)
% Diagonalizes the inputted matrix A using Jacobi rotations
% param A: nxn symmetric matrix
% param eps: scalar tolerance for stopping iteration, compared to off(A)
% return D: near-diagonal nxn matrix whose diagonal elements are A's
% eigenvalues
% return V: nxn orthogonal transformation matrix whose column vectors are
% A's eigenvalues and for which D = V'AV

[m, n] = size(A);
if m~=n || ~isequal(A,A')  % ensures A is square and symmetric
   error('Matrix A must be square and symmetric.');
end

V = eye(n);  % initialization
D = A;
stop = 1/(n-1)*(eps/(2*n - 1))^2;  % stopping value; see theory section of report

max = 1000;
i = 1;
while offset(D) > stop
    [i,j] = get_pivot(D);  % find index of largest abs value element
    R = jacobi(D, i, j);  % gets Jacobi rotation matrix
    D = (R')*D*R;  % applies rotation to A
    V = V*R;  % update eigenvector matrix
    i = i+1;
    if i > max
        disp("Breaking Jacobi Max")
        break
    end
end
    
end