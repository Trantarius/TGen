#include "config.cl"

//handle edge values by normalizing the result
kernel void convolve_normalize(global const FLOAT* input, global FLOAT* output, global const FLOAT* conv, const long conv_size) {
    const long conv_offset = conv_size / 2;

    const long2 pos = (long2)(get_global_id(0),get_global_id(1));
    const long2 off_pos = pos - (long2)(conv_offset,conv_offset);

    FLOAT c=0;
    FLOAT C=0;
    for(long dx=0;dx<conv_size;dx++){
        for(long dy=0;dy<conv_size;dy++){
            if(off_pos.x + dx < 0 || off_pos.x + dx >= get_global_size(0) || 
            off_pos.y + dy < 0 || off_pos.y + dy >= get_global_size(1)){
                continue;
            }
            const long src_lin_id = (off_pos.y + dy) * get_global_size(0) + (off_pos.x + dx);
            const long conv_lin_id = dy * conv_size + dx;
            c += input[src_lin_id] * conv[conv_lin_id];
            C += conv[conv_lin_id];
        }
    }

    const long out_lin_id = pos.y * get_global_size(0) + pos.x;
    output[out_lin_id] = c/C;
}

//handle edges by extending border pixels
kernel void convolve_extend(global const FLOAT* input, global FLOAT* output, global const FLOAT* conv, const long conv_size) {
    const long conv_offset = conv_size / 2;

    const long2 pos = (long2)(get_global_id(0),get_global_id(1));
    const long2 off_pos = pos - (long2)(conv_offset,conv_offset);

    FLOAT c=0;
    for(long dx=0;dx<conv_size;dx++){
        for(long dy=0;dy<conv_size;dy++){
            const long2 src_coord = (long2)(
                min((long)get_global_size(0)-1, max((long)0, off_pos.x + dx)),
                min((long)get_global_size(1)-1, max((long)0, off_pos.y + dy))
            );
            const long src_lin_id = src_coord.y * get_global_size(0) + src_coord.x;
            const long conv_lin_id = dy * conv_size + dx;
            c += input[src_lin_id] * conv[conv_lin_id];
        }
    }

    const long out_lin_id = pos.y * get_global_size(0) + pos.x;
    output[out_lin_id] = c;
}

//handle edges by tiling
kernel void convolve_wrap(global const FLOAT* input, global FLOAT* output, global const FLOAT* conv, const long conv_size) {
    const long conv_offset = conv_size / 2;

    const long2 pos = (long2)(get_global_id(0),get_global_id(1));
    const long2 off_pos = pos - (long2)(conv_offset,conv_offset);

    FLOAT c=0;
    for(long dx=0;dx<conv_size;dx++){
        for(long dy=0;dy<conv_size;dy++){
            const long2 src_coord = (long2)(
                (off_pos.x + dx + get_global_size(0)) % get_global_size(0),
                (off_pos.y + dy + get_global_size(1)) % get_global_size(1)
            );
            const long src_lin_id = src_coord.y * get_global_size(0) + src_coord.x;
            const long conv_lin_id = dy * conv_size + dx;
            c += input[src_lin_id] * conv[conv_lin_id];
        }
    }

    const long out_lin_id = pos.y * get_global_size(0) + pos.x;
    output[out_lin_id] = c;
}

//don't handle edges; input size must be global_size + conv_size - 1
kernel void convolve_shrink(global const FLOAT* input, global FLOAT* output, global const FLOAT* conv, const long conv_size) {

    const long2 pos = (long2)(get_global_id(0),get_global_id(1));
    const long2 in_size = (long2)(get_global_size(0)+conv_size-1,get_global_size(1)+conv_size-1);

    FLOAT c=0;
    for(long dx=0;dx<conv_size;dx++){
        for(long dy=0;dy<conv_size;dy++){
            const long src_lin_id = (pos.y + dy) * in_size.x + (pos.x + dx);
            const long conv_lin_id = dy * conv_size + dx;
            c += input[src_lin_id] * conv[conv_lin_id];
        }
    }

    const long out_lin_id = pos.y * get_global_size(0) + pos.x;
    output[out_lin_id] = c;
}