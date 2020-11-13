function [Vs, Ds]=sorted_eig(A)
% returns the output of eig(A) with eigenvalues and eigenvectors
% sorted from smallest to largest eigenvalue
[V, D] = eig(A);
[~,indices] = sort(diag(D));  % extract eigenvalues and column permutations
Ds = D(indices,indices);  % reorder diagonal elements
Vs = V(:,indices);  % reorder columns
end