function [Q,R]=qr_givens(A)
% Returns the QR decomposition of the square matrix A
% param A: a square matrix (the algorithm could be generalized to mxn
% matrices but I left that out because this project only uses square
% matrices
% return Q: orthogonal transformation matrix such that A = QR
% return R: upper-triangular matrix such that A = QR

n = length(A)
Q=eye(n);
for col=1:n-1
    for row=col+1:n
        if A(row,col)~=0
            a = A(col,col);
            b = A(row, col);
            r=sqrt(a^2+b^2);
            c=a/r;
            s=b/r;
            rot=[c s;-s c];
            A([col,row],col:end)=rot*A([col,row],col:end);
            Q([col,row],:)=rot*Q([col,row],:);
        end
    end
end
R=triu(A);  % A is already tridiagonal, this removes small arithemtic errors from the zero elements
Q=Q';