function Q4=get_Q4(N)
q4_0 = 1:(N);  % main diagonal--delta(i,j) term
q4_0 = 0.75 * (2*(q4_0.^2) + (2*q4_0) + 1);
q4_2 = 1:(N-2);  % second off-diagonals--delta(i,j-2) and (i,j+2) terms
q4_2 = 0.5 * (sqrt((q4_2+1).*(q4_2+2))).*(2*q4_2+3);
q4_4 = 1:(N-4);
q4_4 = 0.25 * sqrt((q4_4+1).*(q4_4+2).*(q4_4+3).*(q4_4+4));
Q4 = diag(q4_0) + diag(q4_2, 2) + diag(q4_2, -2) + diag(q4_4, 4) + diag(q4_4, -4);