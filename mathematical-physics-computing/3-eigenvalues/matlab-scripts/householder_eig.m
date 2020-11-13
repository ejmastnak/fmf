function [P]=householder_eig(A, i, j)
% Creates a Householder reflection using the i,j th element of A
% Used for finding eigenvalues directly with Housereflections
% param A: a nxn symmetric matrix

n = length(A);
v = zeros(n, 1);

v(i:n) = A(i:n,j);
v(i) = v(i) + sign(v(i))*norm(A(i:n,j));
% v(i) = v(i) + sign(v(i))*norm(v)
v = v / norm(v);
P = eye(n) - 2*(kron(v, v'));  % kron is Kronecker product i.e. vv^T 

end