output_dir = "../data/";
output_values = output_dir + "values/";
output_vectors = output_dir + "vectors/";


lambda = [0.0 0.1 0.4 1.0];
N = [100];

eps = 1e-10;
epsJ = 1e-4;


i = 1; % counts total number of iterations in both loops
for n = 1:length(N)
    H0 = get_H0(N(n));
    Q = get_Q4(N(n));
    
    for l = 1:length(lambda)
        H = H0 + lambda(l)*Q;
        
%         % eig
%         [Vref,D] = eig(H);  % use V for reference for sign convention
%         [Vref, D] = sort_eig(Vref, D, Vref);
%         d = diag(D);
%         writematrix(Vref, output_vectors + sprintf("eig-%.1f-%d.csv",lambda(l),N(n)));
%         writematrix(diag(D), output_values + sprintf("eig-%.1f-%d.csv",lambda(l),N(n)));

        % Jacobi max
        [V, D] = eig_jacobi_max(H, epsJ);
        [V, D] = sort_eig(V, D, Vref);
        d = diag(D);
        writematrix(V, output_vectors + sprintf("jac_max-%.1f-%d.csv",lambda(l),N(n)));
        writematrix(diag(D), output_values + sprintf("jac_max-%.1f-%d.csv",lambda(l),N(n)));
        
        
%         Jacobi cyclic
%         [V, D] = eig_jacobi_cyclic(H, epsJ);
%         [V, D] = sort_eig(V, D, Vref);
%         d = diag(D);
%         writematrix(V, output_vectors + sprintf("jac_cyc-%.1f-%d.csv",lambda(l),N(n)));
%         writematrix(diag(D), output_values + sprintf("jac_cyc-%.1f-%d.csv",lambda(l),N(n)));

%         % QR
%         [V, D] = eig_qr(H, eps);
%         [V, D] = sort_eig(V, D, Vref);
%         d = diag(D);
%         writematrix(V, output_vectors + sprintf("qr-%.1f-%d.csv",lambda(l),N(n)));
%         writematrix(diag(D), output_values + sprintf("qr-%.1f-%d.csv",lambda(l),N(n)));

%         % HH with QR
%         [P, T] = trid_householder(H);  % to tridiagonal with HH
%         [V, D] = eig_qr(T, eps);  % complete diagonalization with QR
%         Z = P*V;
%         [V, D] = sort_eig(Z, D, Vref);
%         d = diag(D);
%         writematrix(V, output_vectors + sprintf("HHqr-%.1f-%d.csv",lambda(l),N(n)));
%         writematrix(diag(D), output_values + sprintf("HHqr-%.1f-%d.csv",lambda(l),N(n)));
        
        i = i + 1;
    end
end





% [V1H,D1H] = eig_householder(H1,epsH)
% [V1H,D1H] = sort_eig(V1H,D1H)


% [V1J,D1J] = eig_jacobi(H1,eps);
% [V1J,D1J] = sort_eig(V1J,D1J)

% [V2,D2] = sorted_eig(H2);
% [V4,D4] = sorted_eig(H4);


