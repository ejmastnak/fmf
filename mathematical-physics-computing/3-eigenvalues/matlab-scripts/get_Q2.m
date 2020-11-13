function Q2=get_Q2(N)
q2_0 = 1:(N);  % main diagonal--delta(i,j) term
q2_0 = 0.5 * (q2_0+(q2_0+1));

q2_2 = 1:(N-2);  % second off-diagonals--delta(i,j-2) and (i,j+2) terms
q2_2 = 0.5 * sqrt((q2_2+1).*(q2_2+2));
Q2 = diag(q2_0) + diag(q2_2, 2) + diag(q2_2, -2);

