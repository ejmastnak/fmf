function [P, T]=trid_householder(A)
% Tridiagonalizes the inputted matrix A using Householder transformations
% param A: nxn symmetric matrix
% return T:  a tridiagonal transformation of A
% return P: nxn orhotogonal transformation matrix T = P'AP

N = length(A);

P = eye(N);  % initialize...
T = A;

for row=2:(N-1)
    a = T(row:N,row-1);  % finds the Householder column
    H = eye(N);  % create full-size identiy matrix to hold HH reflection matrix
    H(row:N,row:N) = householder(a);  % get a small householder matrix and embed in larger identity matrix    
    T = H*T*H;  % and not P*T*P' because P = P^T
    P = P*H;
end

end