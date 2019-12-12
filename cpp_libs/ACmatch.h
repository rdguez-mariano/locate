/**
 * @authors Rafael Grompone von Gioi, Mariano Rodríguez
 */

/*----------------------------------------------------------------------------*/

#ifndef _CLIB_ACmatch_H_
#define _CLIB_ACmatch_H_

/*----------------------------------------------------------------------------*/
/** max value */
#define max(a,b) (((a)>(b))?(a):(b))

/** min value */
#define min(a,b) (((a)>(b))?(b):(a))

float * gradient_angle(float * image, int X, int Y, float ** modgrad);

static inline float norm_angle(float a, float b);

float patch_comparison( float * grad_angle1, float * grad_angle2,
                         int X, int Y, float logNT );




void create_weights_for_patch_comparison(int X, int Y);

float weighted_patch_comparison( float * grad_angle1, float * grad_angle2,
                                  float * grad_mod1,   float * grad_mod2,
                                  int X, int Y, float logNT );


float nfa(float logNT, int n, int k, float p);


float quantised_patch_comparison( float * grad_angle1, float * grad_angle2,
                                   float * grad_mod1,   float * grad_mod2,
                                   int X, int Y, float logNT );


#endif

