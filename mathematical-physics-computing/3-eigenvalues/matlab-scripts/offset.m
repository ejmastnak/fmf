function off=offset(A)
% Returns the offset value of the nxn symmetric matrix A
% Offset is the sum of the squares of A's non-diagonal elements

off = 0.0;

n=length(A);
for i=1:n
    for j=1:n
        if i~=j
            off = off + A(i, j)^2;
        end
    end
end

end