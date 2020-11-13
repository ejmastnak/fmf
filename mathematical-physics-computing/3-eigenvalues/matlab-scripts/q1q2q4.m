output_dir = "../data/q1q2q4/";
output_time = output_dir + "times/";
output_values = output_dir + "values/";
output_vectors = output_dir + "vectors/";

lambda = [0.1 0.4 0.7, 1.0];
N = [2000];

% for Q1
tic
for n = 1:length(N)
    H0 = get_H0(N(n));
    Q1 = get_Q1(N(n))^4;
    
    for l = 1:length(lambda)
        H1 = H0 + lambda(l)*Q1;
        [V1,D1] = sorted_eig(H1);
%         writematrix(V1, output_vectors + sprintf("eig-%.1f-%d-H1.csv",lambda(l),N(n)));
%         writematrix(diag(D1), output_values + sprintf("eig-%.1f-%d-H1.csv",lambda(l),N(n)));
    end
end
disp("Q1 time")
toc
disp(" ")


% for Q2
tic
for n = 1:length(N)
    H0 = get_H0(N(n));
    Q2 = get_Q2(N(n))^2;
    
    for l = 1:length(lambda)
        H2 = H0 + lambda(l)*Q2;
        [V2,D2] = sorted_eig(H2);
%         writematrix(V2, output_vectors + sprintf("eig-%.1f-%d-H2.csv",lambda(l),N(n)));
%         writematrix(diag(D2), output_values + sprintf("eig-%.1f-%d-H2.csv",lambda(l),N(n)));
    end
end
disp("Q2 time")
toc
disp(" ")


% for Q4
tic
for n = 1:length(N)
    H0 = get_H0(N(n));
    Q4 = get_Q4(N(n));
    
    for l = 1:length(lambda)
        H4 = H0 + lambda(l)*Q4;
        [V4,D4] = sorted_eig(H4);
%         writematrix(V4, output_vectors + sprintf("eig-%.1f-%d-H4.csv",lambda(l),N(n)));
%         writematrix(diag(D4), output_values + sprintf("eig-%.1f-%d-H4.csv",lambda(l),N(n)));
    end
end
disp("Q4 time")
toc

