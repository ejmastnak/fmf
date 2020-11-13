function[c, s] = givens(a, b)
% From Wikipedia: https://en.wikipedia.org/wiki/Givens_rotation#Triangularization
% Finds c and s
%
if b == 0;
    c = sign(a);
    if (c == 0);
        c = 1.0; % Unlike other languages, MatLab's sign function returns 0 on input 0.
    end;
    s = 0;
elseif a == 0;
    c = 0;
    s = sign(b);
elseif abs(a) > abs(b);
    t = b / a;
    u = sign(a) * sqrt(1 + t * t);
    c = 1 / u;
    s = c * t;
else
    t = a / b;
    u = sign(b) * sqrt(1 + t * t);
    s = 1 / u;
    c = s * t;
end
end