function D=eig_qr_trid(T, eps)
% Illustrates the cubic convergence of tridiagonal QR with Wilkinson's shift
% Adapted from "Applied Numerical Linear Algebra",  Chapter 5
% Written by James Demmel, Nov 13, 1995. Modified Jun  6, 1997
%                 
% param T: symmetric tridiagonal matrix on which to run QR
% param eps:  scalar tolerance for stopping iteration, compared to off(T)
%

n=min(size(T));

%
% Perform QR iteration with Wilkinson's shift
while offset(T) > eps
    % Compute the shift
      lc=T(n-1:n,n-1:n);
      elc=eig(lc);
      if abs(T(n,n)-elc(1)) < abs(T(n,n)-elc(2))
          shift = elc(1);
      else
          shift = elc(2);
      end
    % Perform QR iteration
      [Q,R]=qr(T-shift*eye(n));
      T=R*Q+shift*eye(n);
      T = tril(triu(T,-1),1); % enforce symmetry explicitly
      T = (T+T')/2;
end
D = T;
end
