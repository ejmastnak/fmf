function Q1=get_Q1(N)
q1_1 = 1:(N-1);  % first off diagonals--delta(abs(i-j),1) term
q1_1 = 0.5 * sqrt(q1_1+(q1_1+1)+1); 
Q1 = diag(q1_1, 1) + diag(q1_1, -1);  


