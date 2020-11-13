function [Vs, Ds]=sort_eig(V, D, ref)
% param D: square, diagonal matrix of eigenvalues
% param V: corresponding square matrix of eigenvectors
% param ref: matrix with same eigenvectors as V, but possibly inverted in
% sign
%
% Sorted the matrices into Vs, Ds with eigenvalues/vectors
% sorted from smallest to largest eigenvalue
%
% Then, where necessary, multiplies the eigenvectors in V by -1 to match 
% the sign convention of the eigenvectors in ref


[~,indices] = sort(diag(D));  % extract eigenvalues and column permutations
Ds = D(indices,indices);  % reorder diagonal elements
Vs = V(:,indices);  % reorder columns

for col = 1:length(Vs)
   if sign(Vs(1,col)) ~= sign(ref(1,col))
      Vs(:,col) = -1*Vs(:,col);
   end
end

end