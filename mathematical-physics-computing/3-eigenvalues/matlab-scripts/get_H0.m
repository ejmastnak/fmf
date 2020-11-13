function H0=get_H0(N)
% Returns unperturbed QHO Hamiltonian
h_0=0.5:N;  % generates the vector 0.5, 1.5, ..., N-0.5
H0 = diag(h_0);

