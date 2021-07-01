#ifndef __POINT_H__
#define __POINT_H__

#include "report.h"

typedef float real;
#define real2 float2


#define BLOCK_SIZE 64

struct pointStruct
{
        real x;
        real y;
};

typedef struct pointStruct point;
#endif /* __POINT_H__ */

