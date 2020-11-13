function [i,j]=get_pivot(A)
% Returns indeces (i,j) of the pivot value of the square matrix A, 
% i.e. the largest off-diagonal element by absolute value. 
% Counts indeces from 1, not 0
% Used with my implementation of the Jacobi eigenvalue algorithm---the
% pivot is the optimal element to zero to maximize efficiency

[m, n]=size(A);  % A should be always be square...
i = 1;
j = 2;
max = abs(A(i,j));  % arbitrary initial value at first row, second column
for row=1:m  % loop through each row
    for col=1:n
        if row~=col  % ignore diagonal elements
            temp = abs(A(row, col));
            if temp > max  % update indeces and max value
                max = temp;
                i = row;
                j = col;
            end
        end
    end
end
end