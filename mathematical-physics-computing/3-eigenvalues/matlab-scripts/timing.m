output_dir = "../data/";
output_time = output_dir + "times/";


N = [40, 50, 60, 70];
runs = 5;

eps = 1e-10;
epsJ = 1e-4;

time_length = runs*length(N);
teig = zeros(time_length, 2);
tjac_max = zeros(time_length, 2);
% tjac_cyc = zeros(time_length, 2);
tqr = zeros(time_length, 1);
tHHqr = zeros(time_length, 2);


i = 1; % counts total number of iterations in both loops
for n = 1:length(N)
    H = get_H0(N(n)) + 0.5*get_Q4(N(n));
    
    for run = 1:runs        
        % eig
        tic
        [Vref,D] = eig(H);  % use V for reference for sign convention
        teig(i, 1) = N(n);  % record time
        teig(i, 2) = toc;

        % Jacobi max
        tic
        [V, D] = eig_jacobi_max(H, epsJ);
        tjac_max(i, 1) = N(n);  % record time
        tjac_max(i, 2) = toc;
        
        
%         % Jacobi cyclic
%         tic
%         [V, D] = eig_jacobi_cyclic(H, epsJ);
%         tjac_cyc(i, 1) = N(n);  % record time
%         tjac_cyc(i, 2) = toc;

        % QR
        tic
        [V, D] = eig_qr(H, eps);
        tqr(i, 1) = N(n);  % record time
        tqr(i, 2) = toc;

%         % HH with QR
%         tic
%         [P, T] = trid_householder(H);  % to tridiagonal with HH
%         [V, D] = eig_qr(T, eps);  % complete diagonalization with QR
%         Z = P*V;
%         tHHqr(i, 1) = N(n);  % record time
%         tHHqr(i, 2) = toc;
        
        i = i + 1;
    end
end

% writematrix(teig, output_time + "teig_.csv");
writematrix(tjac_max, output_time + "tjac_max_.csv");
% writematrix(tjac_cyc, output_time + "tjac_cyc_.csv");
% writematrix(tqr, output_time + "tqr_.csv");
% writematrix(tHHqr, output_time + "tHHqr_.csv");




% [V1H,D1H] = eig_householder(H1,epsH)
% [V1H,D1H] = sort_eig(V1H,D1H)


% [V1J,D1J] = eig_jacobi(H1,eps);
% [V1J,D1J] = sort_eig(V1J,D1J)

% [V2,D2] = sorted_eig(H2);
% [V4,D4] = sorted_eig(H4);


