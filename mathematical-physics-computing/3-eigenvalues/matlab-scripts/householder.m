function P=householder(a)
% Creates a Householder reflection P corresponding to the vector a
% param a: a (nx1) vector 
% return P: and (nxn) householder matrix defined by a
n = length(a);
v = a;  % initialize vector defining the reflection
v(1) = v(1) + sign(v(1))*norm(a);  % add the projection term to first component
v = v/norm(v);  % normalize v
P = eye(n) - 2*(kron(v, v'));  % kron is Kronecker product i.e. vv^T 
end