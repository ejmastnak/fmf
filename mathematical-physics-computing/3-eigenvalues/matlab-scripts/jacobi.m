function [R]=jacobi(A, i, j)
% Creates a Jacobi rotation matrix R_{ij} corresponding to the 
% (i,j)th element of the matrix A
% param A: an nxn square symmetric matrix

[m, n] = size(A);
if m~=n  % ensures A is square
   error('Matrix A must be square.');
end

phi = (abs(A(j, j) - A(i, i)))/(2*A(i, j));
t = 1.0/(abs(phi) + sqrt(phi^2 + 1.0));
c = 1.0/sqrt(t^2 + 1.0);
s = t*c;

R = eye(m);
R(i, i) = c;
R(j, j) = c;
R(i, j) = s;
R(j, i) = -s;

end