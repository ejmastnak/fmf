function [c,s]=givens(a, b)
% Returns the [c, s] terms for a Givens rotation matrix
% corresponding to the terms a, b
r=sqrt(a^2+b^2);
c=a/r;
s= b/r;

 