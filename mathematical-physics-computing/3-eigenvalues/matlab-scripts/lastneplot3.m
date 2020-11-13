
n = 10;

%generacija operatorjev
h=(1/2):n;
H0 = diag(h);

q = 1:(n-1);
q = q+(q+1)+1;
q = sqrt(q);
q = (1/2)*q;

Q1 = diag(q, 1) + diag(q, -1);

q21 = 1:(n);
q21 = q21+(q21+1);
q21 = (1/2)*q21;

q22 = 1:(n-2);
q22 = (q22+1).*(q22+2);
q22 = sqrt(q22);
q22 = (1/2)*q22;

Q2 = diag(q21) + diag(q22, 2) + diag(q22, -2);

q41 = 1:(n);
q41 = (2*(q41.*q41) + (2*q41)+1);
q41 = (3/4)*q41;

q42 = 1:(n-2);
q42 =(sqrt((q42+1).*(q42+2))).*(2*q42+3);
q42 = (1/2)*q42;

q44 = 1:(n-4);
q44 = sqrt((q44+1).*(q44+2).*(q44+3).*(q44+4));
q44 = (1/4)*q44;

Q4 = diag(q41) + diag(q42, 2) + diag(q42, -2) + diag(q44, 4) + diag(q44, -4);

H = H0 - (5/2)*Q2 + (1/10)*(Q4)

%izracun referencnih matlabovih lastnih vrednosti
[R, D] = eig(H);
d = diag(D);
    
%lastne plot
q = -4.5 : 0.01 : 4.5;
v2 = -2*(q.^2) + (q.^4)/10;
[k, l] = size(q);

lastna = zeros(10, l);
for i=1:l
    for j=1:10
        lastna(j, i) = d(j); 
    end
end
vektorji = zeros(10, n);


% for i=0:9
%    for j=0:99
%        if R(i+1,j+1) > 0.0001
%             a = (exp(-q.^2/2));
%             b = hermiteH(j, q);
%             lastna(i+1,:) = lastna(i+1,:)+(R(i+1,j+1)*(((2^j)*factorial(j)*sqrt(pi))^(-1/2))*a.*b);
%             vektorji(i+1, j+1) = R(i+1,j+1);
%        end
%    end 
%    plot(q, lastna(i+1,:)); hold on;
% end    

plot(q, v2);
title('Lastne funkcije s potencialom');
xlabel('q');
ylabel('E(n)');



