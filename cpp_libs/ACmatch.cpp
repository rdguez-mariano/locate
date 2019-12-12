/**
 * @authors Rafael Grompone von Gioi, Mariano Rodríguez
 */

/*----------------------------------------------------------------------------*/
#include "ACmatch.h"


#include <iostream>
#include <stdio.h>
#include "math.h"



#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/*----------------------------------------------------------------------------*/
/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */

/*----------------------------------------------------------------------------*/

#define NOTDEF -1024.0



float* weights;
float sum_log_w;
float threshold_AC;
float sigma_default = -1.0;

int NewOriSize1 = 60;
float quant_prec = 0.032;
float step_sigma = -1.0;


void * xmalloc(size_t size)
{
  void * p;
  if( size == 0 ) std::cerr<<"xmalloc: zero size"<<std::endl;
  p = malloc(size);
  if( p == NULL ) std::cerr<<"xmalloc: out of memory"<<std::endl;
  return p;
}




float * gradient_angle(float * image, int X, int Y, float ** modgrad)
{
  float * grad_angle;
  float * grad_mod;
  float dx,dy;
  int x,y;

  /* get memory */
  grad_angle = (float *) xmalloc( X * Y * sizeof(float) );
  grad_mod   = (float *) xmalloc( X * Y * sizeof(float) );

  
  /* edges have not gradient defined */
  for(x=0; x<X; x++) grad_angle[x+0*X] = grad_angle[x+(Y-1)*X] = NOTDEF;
  for(y=0; y<Y; y++) grad_angle[0+y*X] = grad_angle[X-1+y*X]   = NOTDEF;
  for(x=0; x<X; x++) grad_mod[x+0*X]   = grad_mod[x+(Y-1)*X]   = 0.0;
  for(y=0; y<Y; y++) grad_mod[0+y*X]   = grad_mod[X-1+y*X]     = 0.0;
  
  /* process */
  for(x=1; x<X-1; x++)
    for(y=1; y<Y-1; y++)
      {
        dx = 0.5 * (image[(x+1)+y*X] - image[(x-1)+y*X]);
        dy = 0.5 * (image[x+(y+1)*X] - image[x+(y-1)*X]);

        grad_mod[x+y*X] = sqrt(dx*dx + dy*dy);

        if((image[(x+1)+y*X]==NOTDEF)||(image[(x-1)+y*X]==NOTDEF)||(image[x+(y+1)*X]==NOTDEF)||(image[x+(y-1)*X]==NOTDEF))
        {
            grad_angle[x+y*X] = NOTDEF;
            grad_mod[x+y*X] = 0;
        }
        else
        // if( grad_mod[x+y*X] < 3.0 ) grad_angle[x+y*X] = NOTDEF;
        if( grad_mod[x+y*X] == (float)0 ) grad_angle[x+y*X] = NOTDEF;
        else
                grad_angle[x+y*X] = atan2(dy,dx);
      }
      
  /* return values */
//   *modgrad = grad_mod;
  return grad_angle;
}


/*----------------------------------------------------------------------------*/

static inline float norm_angle(float a, float b)
{
    a -= b;
    while( a <= -M_PI ) a += 2.0*M_PI;
    while( a >   M_PI ) a -= 2.0*M_PI;

    return fabs(a) / M_PI;
}

/*----------------------------------------------------------------------------*/

float patch_comparison( float * grad_angle1, float * grad_angle2,
                         int X, int Y, float logNT )
{
    int n = 0;      /* count of angles compared */
    float k = 0.0; /* measure of symmetric angles */
    float logNFAC;

    int nmax = (X-2)*(Y-2);
    /* logNFAC-logNT <= 0 */
    /* logNFAC-logNT = nmax * log10(k) - 0.5 * log10(2.0 * M_PI) - (nmax+0.5) * log10(nmax) + nmax * log10(exp(1.0)) */
    float threshold = pow(10, (0.5 * log10(2.0 * M_PI) + (nmax+0.5) * log10(nmax) - nmax * log10(exp(1.0)))/nmax );
    int x,y,A,B;


    for(x=1; x<X-1; x++)
        for(y=1; y<Y-1; y++)
        {
            A = grad_angle1[x+y*X] > NOTDEF;  /* A has defined gradient */
            B = grad_angle2[x+y*X] > NOTDEF;  /* B has defined gradient */

            /* if at least one of the corresponding pixels has defined gradient,
           count it in the total number of pixels evaluated */
            if( A || B ) ++n;

            if( A && B) k += norm_angle(grad_angle1[x+y*X], grad_angle2[x+y*X]);
            else if( (A && !B) || (!A && B) ) k += 1.0;

            if (k>threshold)
                return (logNT);
        }

    /* NFAC = NT * k^n / n!
     log(n!) is bounded by Stirling's approximation:
       n! >= sqrt(2pi) * n^(n+0.5) * exp(-n)
     then, log10(NFA) <= log10(NT) + n*log10(k) - log10(latter expansion) */
    logNFAC = logNT + n * log10(k)
            - 0.5 * log10(2.0 * M_PI) - (n+0.5) * log10(n) + n * log10(exp(1.0));

    return logNFAC;
}


/*----------------------------------------------------------------------------*/

void create_weights_for_patch_comparison(int X, int Y)
{
    sum_log_w = 0.0;
    delete[] weights;
    weights = new float[X*Y];
    int r = (int) (X/2), c = (int) (Y/2);
    float w;

    for(int x=0; x<X*Y; x++)
        weights[x] = NOTDEF;

    for(int x=1; x<X-1; x++)
        for(int y=1; y<Y-1; y++)
        {
            //w = exp( -(pow(r-x,2)+pow(c-y,2))/(2.0*X*sqrt(X)) );
            if (sigma_default>0)
                w = exp( -(pow(r-x,2)+pow(c-y,2))/(2.0*sigma_default) );
            else
                w = exp( -(pow(r-x,2)+pow(c-y,2))/(2.0*X*Y) );
            weights[x+y*X] = w;
            sum_log_w += log10(w);
        }
    int nmax = (X-2)*(Y-2);
    /* logNFAC-logNT <= 0 */
    /* logNFAC-logNT = nmax * log10(k) - 0.5 * log10(2.0 * M_PI) - (nmax+0.5) * log10(nmax) + nmax * log10(exp(1.0)) */
    threshold_AC = pow(10, (sum_log_w + 0.5 * log10(2.0 * M_PI) + (nmax+0.5) * log10(nmax) - nmax * log10(exp(1.0)))/nmax );
}

float weighted_patch_comparison( float * grad_angle1, float * grad_angle2,
                                  float * grad_mod1,   float * grad_mod2,
                                  int X, int Y, float logNT )
{
    int n = 0;      /* count of angles compared */
    float k = 0.0; /* measure of symmetric angles */
    float logNFAC;
    int x,y;

    for(x=1; x<X-1; x++)
        for(y=1; y<Y-1; y++)
        {
            float a = grad_angle1[x+y*X];
            float b = grad_angle2[x+y*X];
            int A = (a != NOTDEF);  /* A has defined gradient */
            int B = (b != NOTDEF);  /* B has defined gradient */

            /* if at least one of the corresponding pixels has defined gradient,
           count it in the total number of pixels evaluated */
            if( A || B )
            {
                ++n;
                if( A && B) k += weights[x+y*X] * norm_angle(a,b);
                else        k += weights[x+y*X];  /* one angle not defined, maximal error = 1 */
            }

            if (k>threshold_AC)
                return (logNT);
        }

    /* NFAC = NT * k^n / (n! * prod_i w_i)
     log(n!) is bounded by Stirling's approximation:
       n! >= sqrt(2pi) * n^(n+0.5) * exp(-n)
     then, log10(NFA) <= log10(NT) + n*log10(k) - log10(latter expansion) */
    logNFAC = logNT + n * log10(k) - sum_log_w - 0.5 * log10(2.0 * M_PI) - (n+0.5) * log10(n) + n * log10(exp(1.0));

    return logNFAC;
}



/*----------------------------------------------------------------------------*/

float nfa(float logNT, int n, int k, float p)
{
    float r = (float) k / (float) n;

    if( r <= p ) return logNT;

    float log_binom = k * log10(p/r) + n*(1-r) * log10( (1-p)/(1-r) );
    return logNT + log_binom;
}


float quantised_patch_comparison( float * grad_angle1, float * grad_angle2,
                                   float * grad_mod1,   float * grad_mod2,
                                   int X, int Y, float logNT )
{
    int n = 0;      /* count of angles compared */
    int k = 0; /* measure of symmetric angles */
    float logNFAC;
    int x,y;


    for(x=1; x<X-1; x++)
        for(y=1; y<Y-1; y++)
        {
            float a = grad_angle1[x+y*X];
            float b = grad_angle2[x+y*X];
            int A = (a != NOTDEF);  /* A has defined gradient */
            int B = (b != NOTDEF);  /* B has defined gradient */

            /* if at least one of the corresponding pixels has defined gradient,
           count it in the total number of pixels evaluated */
            if( A || B ) //if( A || B )
            {
                ++n;

                if(( A && B)&&(norm_angle(a,b)<quant_prec))
                    k ++ ;
                /* one angle not defined, maximal error = 1 */
            }
        }

    /* NFAC = NT * k^n / (n! * prod_i w_i)
     log(n!) is bounded by Stirling's approximation:
       n! >= sqrt(2pi) * n^(n+0.5) * exp(-n)
     then, log10(NFA) <= log10(NT) + n*log10(k) - log10(latter expansion) */

    logNFAC = nfa(logNT,n,k,quant_prec);

    return logNFAC;
}


