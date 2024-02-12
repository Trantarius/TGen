
#define PASTE2(LEFT,RIGHT) LEFT##RIGHT
#define PASTE(LEFT,RIGHT) PASTE2(LEFT,RIGHT)

#ifndef FLOAT
#define FLOAT float
#endif

#define FLOAT2 PASTE(FLOAT,2)
#define FLOAT3 PASTE(FLOAT,3)
#define FLOAT4 PASTE(FLOAT,4)

#ifndef SEDIMENT_CAPACITY
#define SEDIMENT_CAPACITY ((FLOAT)1.0)
#endif

#ifndef EROSION_SPEED
#define EROSION_SPEED ((FLOAT)0.05)
#endif

#ifndef MAX_NOISE_FREQ
#define MAX_NOISE_FREQ ((FLOAT)0.25)
#endif

#ifndef NOISE_FUNCTION
#define NOISE_FUNCTION valuenoise3
#endif