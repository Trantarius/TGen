
#include "opensimplex.cl"
#include "convolve.cl"
#include "config.cl"

long get_land_id(long2 coord){
    return coord.y * get_global_size(0) + coord.x;
}

long get_water_id(long2 coord){
    return get_global_size(0)*get_global_size(1) + get_land_id(coord);
}

kernel void map_range(global FLOAT* field, const FLOAT src_low, const FLOAT src_high, 
const FLOAT dst_low, const FLOAT dst_high){
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    field[lin_id] = (field[lin_id]-src_low)/(src_high-src_low) * (dst_high-dst_low) + dst_low;
}

kernel void to_int32(global const FLOAT* field, global int* output){
    long lin_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    output[lin_id] = (int)(field[lin_id] * (FLOAT)0xffff);
}

kernel void flatten_slopes(global const FLOAT* input, global FLOAT* output, const FLOAT amount){
    const long2 coord = (long2)(get_global_id(0), get_global_id(1));
    const long2 size = (long2)(get_global_size(0), get_global_size(1));
    const long id = coord.y * size.x + coord.x;

    FLOAT min=INFINITY;
    FLOAT max=-INFINITY;

    for(long dx=-1;dx<=1;dx++){
        for(long dy=-1;dy<=1;dy++){
            long2 h = coord + (long2)(dx,dy);
            if(h.x<0 || h.y<0 || h.x>=size.x || h.y>=size.y || (dx==0&&dy==0)){
                continue;
            }
            long h_id = h.y * size.x + h.x;
            FLOAT h_v = (input[h_id]-input[id])/length((FLOAT2)(dx,dy));
            if(h_v<min){
                min=h_v;
            }
            else if(h_v>max){
                max=h_v;
            }
        }
    }

    if(min==INFINITY || max==-INFINITY){
        output[id]=input[id];
        return;
    }

    output[id] = input[id] + amount * (min+max)/2.0;
}