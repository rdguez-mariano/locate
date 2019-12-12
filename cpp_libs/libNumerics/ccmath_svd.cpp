/**
 * @file ccmath_svd.cpp
 * @brief SVD (from ccmath 2.2.1)
 * @author Daniel A. Atkinson, Pascal Monasse
 * 
 * Copyright (c) 2001, 2012 Daniel A. Atkinson <DanAtk@aol.com>, Pascal Monasse
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// File extracted from ccmath 2.2.1 (http://freecode.com/projects/ccmath)
// Changes Pascal Monasse:
// -template (instead of double)
// -remove svduv (used normally for 2m<=3n) as it has bugs
// -protect from erroneous code when m<n
// -protect from dramatic cancellation in qrbdv
// -increase number of iterations in qrbdv (200->5000)

#ifdef MATRIX_H // Do nothing if not included from matrix.h

#include <cmath>
#include <cstdlib>
#include <cassert>

// (auxiliary function)
// Perform a QR reduction of a real bidiagonal matrix and update the
// orthogonal transformation matrices U and V.
template <typename T>
static int qrbdv(T *dm,T *em,T *um,int mm,T *vm,int m)
{ int i,j,k,n,jj,nm;
  T u,x,y,a,b,c,s,t,w,*p,*q;
  for (j=1,t=fabs(dm[0]); j<m ;++j)
    if((s=fabs(dm[j])+fabs(em[j-1]))>t) t=s;
  t*=1.e-15; n=5000*m; nm=m;
  for(j=0; m>1 && j<n ;++j){
    for(k=m-1; k>0 ;--k){
      if(fabs(em[k-1])<t) break;
      if(fabs(dm[k-1])<t){
        for(i=k,s=1.,c=0.; i<m ;++i){
          a=s*em[i-1]; b=dm[i]; em[i-1]*=c;
          dm[i]=u=sqrt(a*a+b*b); s= -a/u; c=b/u;
          for(jj=0,p=um+k-1; jj<mm ;++jj,p+=mm){
            q=p+i-k+1;
            w=c* *p+s* *q; *q=c* *q-s* *p; *p=w;
           }
	     }
        break;
       }
     }
    y=dm[k]; x=dm[m-1]; u=em[m-2];
    a=(y+x)*(y-x)-u*u; s=y*em[k]; b=s+s;
    u=sqrt(a*a+b*b);
    if(u!=0.){
      if(a >=0)
          a+= u;
      else // Protect against dramatic cancellation
          a = (b*b)/(u-a);
      c=sqrt(a/(u+u));
	  if(c!=0.) s/=(c*u); else s=1.;
	  for(i=k; i<m-1 ;++i){
        b=em[i];
        if(i>k){
          a=s*em[i]; b*=c;
	      em[i-1]=u=sqrt(x*x+a*a);
    	  c=x/u; s=a/u;
         }
        a=c*y+s*b; b=c*b-s*y;
        for(jj=0,p=vm+i; jj<nm ;++jj,p+=nm){
          w=c* *p+s* *(p+1); *(p+1)=c* *(p+1)-s* *p; *p=w;
         }
        s*=dm[i+1]; dm[i]=u=sqrt(a*a+s*s);
        y=c*dm[i+1]; c=a/u; s/=u;
        x=c*b+s*y; y=c*y-s*b;
        for(jj=0,p=um+i; jj<mm ;++jj,p+=mm){
          w=c* *p+s* *(p+1); *(p+1)=c* *(p+1)-s* *p; *p=w;
         }
	   }
     }
    em[m-2]=x; dm[m-1]=y;
    if(fabs(x)<t) --m;
    if(m==k+1) --m;
   }
  return j;
}

// (auxiliary function)
// Perform a left Householder transform matrix U from the vectors
// specifying the Householder reflections.
template <typename T>
static void ldumat(T *a,T *u,int m,int n)
{ T *p0,*q0,*p,*q,*w;
  int i,j,k,mm;
  T s,h;
  w=(T *)calloc(m,sizeof(T));
  for(i=0,mm=m*m,q=u; i<mm ;++i) *q++ =0.;
  p0=a+n*n-1; q0=u+m*m-1; mm=m-n; i=n-1;
  for(j=0; j<mm ;++j,q0-=m+1) *q0=1.;
  if(mm==0){ p0-=n+1; *q0=1.; q0-=m+1; --i; ++mm;}
  for(; i>=0 ;--i,++mm,p0-=n+1,q0-=m+1){
    if(*p0!=0.){
      for(j=0,p=p0+n,h=1.; j<mm ;p+=n) w[j++]= *p;
      h= *p0; *q0=1.-h;
      for(j=0,q=q0+m; j<mm ;q+=m) *q= -h*w[j++];
      for(k=i+1,q=q0+1; k<m ;++k){
	for(j=0,p=q+m,s=0.; j<mm ;p+=m) s+=w[j++]* *p;
	s*=h;
	for(j=0,p=q+m; j<mm ;p+=m) *p-=s*w[j++];
        *q++ = -s;
       }
     }
    else{
      *q0=1.;
      for(j=0,p=q0+1,q=q0+m; j<mm ;++j,q+=m) *q= *p++ =0.;
     }
   }
  free(w);
}

// (auxiliary function)
// Perform a right Householder transform matrix U from the vectors
// specifying the Householder reflections.
template <typename T>
static void ldvmat(T *a,T *v,int n)
{ T *p0,*q0,*p,*q,*qq;
  T h,s;
  int i,j,k,mm;
  for(i=0,mm=n*n,q=v; i<mm ;++i) *q++ =0.;
  *v=1.; q0=v+n*n-1; *q0=1.; q0-=n+1;
  p0=a+n*n-n-n-1;
  for(i=n-2,mm=1; i>0 ;--i,p0-=n+1,q0-=n+1,++mm){
    if(*(p0-1)!=0.){
      for(j=0,p=p0,h=1.; j<mm ;++j,++p) h+= *p* *p;
      h= *(p0-1); *q0=1.-h;
      for(j=0,q=q0+n,p=p0; j<mm ;++j,q+=n) *q= -h* *p++; 
      for(k=i+1,q=q0+1; k<n ;++k){
        for(j=0,qq=q+n,p=p0,s=0.; j<mm ;++j,qq+=n) s+= *qq* *p++;
        s*=h;
        for(j=0,qq=q+n,p=p0; j<mm ;++j,qq+=n) *qq-=s* *p++;
        *q++ = -s;
       }
     }
    else{
      *q0=1.;
      for(j=0,p=q0+1,q=q0+n; j<mm ;++j,q+=n) *q= *p++ =0.;
     }
   }
}

// Compute the singular value decomposition A=U*D*V'
// (efficient when m is much larger than n)
template <typename T>
int sv2uv(T *d,T *a,T *u,int m,T *v,int n)
{ T *p,*p1,*q,*pp,*w,*e;
  T s,t,h,r,sv;
  int i,j,k,mm,nm,ms;
  if(m<n) return -1;
  w=(T *)calloc(m+n,sizeof(T)); e=w+m;
  for(i=0,mm=m,p=a; i<n ;++i,--mm,p+=n+1){
    if(mm>1){ sv=h=0.;
      for(j=0,q=p,s=0.; j<mm ;++j,q+=n){
	w[j]= *q; s+= *q* *q;
       }
      if(s>0.){
	h=sqrt(s); if(*p<0.) h= -h;
	s+= *p*h; s=1./s; t=1./(w[0]+=h);
        sv=1.+fabs(*p/h);
	for(k=1,ms=n-i; k<ms ;++k){
	  for(j=0,q=p+k,r=0.; j<mm ;q+=n) r+=w[j++]* *q;
	  r=r*s;
	  for(j=0,q=p+k; j<mm ;q+=n) *q-=r*w[j++];
	 }
	for(j=1,q=p; j<mm ;) *(q+=n)=w[j++]*t;
       }
      *p=sv; d[i]= -h;
     }
    if(mm==1) d[i]= *p;
   }
  ldumat(a,u,m,n);
  for(i=0,q=a; i<n ;++i){
    for(j=0; j<n ;++j,++q){
      if(j<i) *q=0.;
      else if(j==i) *q=d[i];
     }
   }
  for(i=0,mm=n,nm=n-1,p=a; i<n ;++i,--mm,--nm,p+=n+1){
    if(i && mm>1){ sv=h=0.;
      for(j=0,q=p,s=0.; j<mm ;++j,q+=n){
	w[j]= *q; s+= *q* *q;
       }
      if(s>0.){
	h=sqrt(s); if(*p<0.) h= -h;
	s+= *p*h; s=1./s; t=1./(w[0]+=h);
        sv=1.+fabs(*p/h);
	for(k=1,ms=n-i; k<ms ;++k){
	  for(j=0,q=p+k,r=0.; j<mm ;q+=n) r+=w[j++]* *q;
	  for(j=0,q=p+k,r*=s; j<mm ;q+=n) *q-=r*w[j++];
	 }
        for(k=0,p1=u+i; k<m ;++k,p1+=m){
          for(j=0,q=p1,r=0.; j<mm ;) r+=w[j++]* *q++;
	  for(j=0,q=p1,r*=s; j<mm ;) *q++ -=r*w[j++];
	 }
       }
      *p=sv; d[i]= -h;
     }
    if(mm==1) d[i]= *p;
    p1=p+1;
    if(nm>1){ sv=h=0.;
      for(j=0,q=p1,s=0.; j<nm ;++j,++q) s+= *q* *q;
      if(s>0.){
	h=sqrt(s); if(*p1<0.) h= -h;
        sv=1.+fabs(*p1/h);
	s+= *p1*h; s=1./s; t=1./(*p1+=h);
	for(k=n,ms=n*(n-i); k<ms ;k+=n){
	  for(j=0,q=p1,pp=p1+k,r=0.; j<nm ;++j) r+= *q++ * *pp++;
	  for(j=0,q=p1,pp=p1+k,r*=s; j<nm ;++j) *pp++ -=r* *q++;
	 }
	for(j=1,q=p1+1; j<nm ;++j) *q++ *=t;
       }
      *p1=sv; e[i]= -h;
     }
    if(nm==1) e[i]= *p1;
   }
  ldvmat(a,v,n);
  qrbdv(d,e,u,m,v,n);
  for(i=0; i<n ;++i){
    if(d[i]<0.){ d[i]= -d[i];
      for(j=0,p=v+i; j<n ;++j,p+=n) *p= - *p;
     }
   }
  free(w);
  return 0;
}

// Compute the singular value decomposition A=U*D*V'
// (calls the apropriate function above)
template <typename T>
int svd(T *d, T *a, T *u, int m, T *v, int n)
{
    assert(m>=n);
    return sv2uv(d, a, u, m, v, n);
}

#endif // MATRIX_H
