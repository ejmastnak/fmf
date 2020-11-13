N = 8;
lambda = 0.8;
epsJ = 1e-4;
eps = 1e-10;

A = tril(magic(N));
A = A + A';

a = 2;
b = 3;
t = zeros(a*b, 2);
it = 1;
for i = 1:a
    for j = 1:b
        t(it, 1) = i;
        t(it, 2) = -1;
        it = it + 1;
    end
end
t








% [P, T] = trid_householder(A);
% [Q, R] = qr_givens(A);

% [V, D] = sorted_eig(A)
% [VJ, DJ] = eig_jacobi_max(A, eps);
% [VH, DH] = eig_householder(A, eps);