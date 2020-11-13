# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import math
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pylab import *
import sys
import time

def p1(z):
    nf=1
    x=z
    x1=0
    s=[]
    zz=z

    for i in range(10):
        if i>0:
            nf=nf*i
            zz=zz*z**2
        if i%2==0:
            a=1
        else: a=-1

        x=a*(zz/(nf*(2*i+1)))
        x1=x1+x
        s=s+[x1]
    return 2/(math.sqrt(math.pi))*x1

def p2(z):
    x1=1-(p1(z))
    return x1

def a1(z):
    if z==0:
        return 0
    gor=1
    dol=(-2*z**2)
    x=0
    k=1
    k1=0
    while abs(k)>abs(k1):
        gor1=gor*(2*i-1)
        dol1=dol*(-2*z**2)
        k=gor/dol
        k1=gor1/dol1
        x=x+gor/dol
    erfc= (1+x)/(z*math.sqrt(math.pi)*math.exp(z**2))
    return 1-erfc

def a2(z):
    if z==0:
        return 0
    gor=1
    dol=(-2*z**2)
    x=0
    k=1
    k1=0
    while abs(k)>abs(k1):
        gor1=gor*(2*i-1)
        dol1=dol*(-2*z**2)
        k=gor/dol
        k1=gor1/dol1
        x=x+gor/dol
    erfc= (1+x)/(z*math.sqrt(math.pi)*math.exp(z**2))
    return erfc

def r1(z):
    p=0.3275911
    a=0.254829592
    b=-0.284496736
    c=1.421413741
    d=-1.45315202
    e=1.061405429
    t=1/(1+p*z)
    erf=1-(a*t+b*t**2+c*t**3+d*t**4+e*t**5)*math.exp(-z**2)
    return erf

def r2(z):
    p=0.3275911
    a=0.254829592
    b=-0.284496736
    c=1.421413741
    d=-1.45315202
    e=1.061405429
    t=1/(1+p*z)
    erf=1-(a*t+b*t**2+c*t**3+d*t**4+e*t**5)*math.exp(-z**2)
    return 1-erf

tabela1=[]
tabela2=[]
tabela3=[]

for i in range(0,32,2):
    tabela1=tabela1+[[p1(i/10),a1(i/10),r1(i/10),math.erf(i/10)]]
for i in range(30,85,5):
    tabela2=tabela2+[[p2(i/10),a2(i/10),r2(i/10),math.erfc(i/10)]]
for i in range(30,85,5):
    tabela3=tabela3+[[p1(i/10),a1(i/10),r1(i/10),math.erf(i/10)]]

# print(np.array(tabela1))
# print(np.array(tabela2))

napake1=[]
for i in range(1,len(tabela1)):
    napake1=napake1+[[abs(tabela1[i][0]-tabela1[i][3])/tabela1[i][3],abs(tabela1[i][1]-tabela1[i][3])/tabela1[i][3],abs(tabela1[i][2]-tabela1[i][3])/tabela1[i][3]]]
napake2=[]
for i in range(len(tabela3)):
    napake2=napake2+[[abs(tabela3[i][0]-tabela3[i][3])/tabela3[i][3],abs(tabela3[i][1]-tabela3[i][3])/tabela3[i][3],abs(tabela3[i][2]-tabela3[i][3])/tabela3[i][3]]]

napake1_pot=[]
for i in range(len(napake1)):
    napake1_pot=napake1_pot+[napake1[i][0]]
napake1_asi=[]
for i in range(len(napake1)):
    napake1_asi=napake1_asi+[napake1[i][1]]
napake1_rac=[]
for i in range(len(napake1)):
    napake1_rac=napake1_rac+[napake1[i][2]]

p=plt.plot(np.arange(0.2,3.2,0.2),np.array(napake1_pot),'ko')
a=plt.plot(np.arange(0.2,3.2,0.2),np.array(napake1_asi),'rv')
r=plt.plot(np.arange(0.2,3.2,0.2),np.array(napake1_rac),'gs')
plt.legend(('potencialna','asimptotska','racionalna'),loc='lower right')
plt.yscale('log')
plt.xlabel('z')
plt.ylabel('error w.r.t. math.erf(z)')
plt.show()


napake2_pot=[]
for i in range(len(napake2)):
    napake2_pot=napake2_pot+[napake2[i][0]]
napake2_asi=[]
for i in range(len(napake2)):
    napake2_asi=napake2_asi+[napake2[i][1]]
napake2_rac=[]
for i in range(len(napake2)):
    napake2_rac=napake2_rac+[napake2[i][2]]


p22=plt.plot(np.arange(3,8.5,0.5),np.array(napake2_pot),'ko')
a22=plt.plot(np.arange(3,8.5,0.5),np.array(napake2_asi),'rv')
r22=plt.plot(np.arange(3,8.5,0.5),np.array(napake2_rac),'gs')
plt.legend(('potencialna','asimptotska','racionalna'),loc='lower right')
plt.yscale('log')
plt.xlabel('z')
plt.ylabel('error w.r.t. math.erf(z)')
plt.show()

prava1=[]
for i in range(12):
    prava1=prava1+[tabela1[i][3]]
prava2=[]
for i in range(5,11):
    prava2=prava2+[tabela2[i][3]]


pot1=[]
for i in range(12):
    pot1=pot1+[tabela1[i][0]]


asi2=[]
for i in range(5,11):
    asi2=asi2+[tabela2[i][1]]

rac1=[]
for i in range(len(tabela1)):
    rac1=rac1+[tabela1[i][0]]
rac2=[]
for i in range(len(tabela2)):
    rac2=rac2+[tabela2[i][0]]

def ak(z):
    x=-1/(2*z**2)
    nf=1
    x=0
    tabela3=[]
    for i in range(0,50):
        nf=nf*(2*i-1)
        x=nf/((-2*z**2)**i)
        tabela3=tabela3+[abs(x)]
    return tabela3



plt.plot(np.array(ak(3)),'ko',np.array(ak(4)),'ro',np.array(ak(5)),'go')
plt.legend(('asympt z=3','z=4','z=5'),loc='lower right')
plt.yscale('log')
plt.xlabel('number of terms in asymp')
plt.ylabel('error w.r.t. math.erf(z)')
plt.show()


def er(z):
    if z<1:
        return p1(z)
    elif z>=1 and z<4:
        return r1(z)
    else:
        return a1(z)

err=[]
xerr=[]
for i in range(0,80,2):
    xerr=xerr+[i/10]
    err=err+[er(i/10)]
erp=[]
xerp=[]
for i in range(0,80,2):
    xerp=xerp+[i/10]
    erp=erp+[math.erf(i/10)]

plt.plot(np.array(xerp),np.array(erp),'ko',np.array(xerr),np.array(err))
plt.legend(('Vgrajen erf','zlepek'),loc='lower right')
plt.xlabel('z')
plt.ylabel('erf(z)')
plt.show()


def f(x):
    return math.exp(-x**2)

def simpson(f,n,k):
    z=0
    h=k/n

    fun=0


    for i in range(int(n/2)):
        fun=fun+(1/3*h*(f(z)+4*f(z+h)+f(z+2*h)))

        z=z+2*h

    return 2/math.sqrt(math.pi)*fun

def sim(z):
    return simpson(f,10,z)

ers=[]
xers=[]
for i in range(0,80,2):
    xers=xers+[i/10]
    ers=ers+[simpson(f,10,i/10)]

plt.plot(np.array(xers),np.array(ers),'ko',np.array(xerr),np.array(err),'ro')
plt.legend(('Simpson','zlepek'),loc='lower right')
plt.xlabel('z')
plt.ylabel('erf(z)')
plt.show()
