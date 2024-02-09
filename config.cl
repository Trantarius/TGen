
#define PASTE2(LEFT,RIGHT) LEFT##RIGHT
#define PASTE(LEFT,RIGHT) PASTE2(LEFT,RIGHT)

#ifndef FLOAT
#define FLOAT float
#endif

//#define FLOAT float
#define FLOAT2 PASTE(FLOAT,2)
#define FLOAT3 PASTE(FLOAT,3)
#define FLOAT4 PASTE(FLOAT,4)

#ifndef IMG_SIZE
#define IMG_SIZE 256
#endif