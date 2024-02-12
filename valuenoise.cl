#include "config.cl"

ulong xorshift64(ulong x){
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
}

FLOAT cell_rand(const ulong seed, const long3 coord){
    ulong x = (coord.x&0xfffffUL) ^ ((ulong)(coord.y&0xfffffUL)<<20UL) ^ ((ulong)(coord.z&0xfffffUL)<<40UL);
    x ^= seed;
    x = xorshift64(x);
    FLOAT ret = (double)(x%0xc9c21056da55bfaUL) / (double)0xc9c21056da55bfaUL;
    return ret;
}

FLOAT valuenoise3(const ulong seed, FLOAT3 pos){

    pos = (FLOAT3)(
        dot(pos,(FLOAT3)(0.211325, -0.788675, 0.577350)),
        dot(pos,(FLOAT3)(-0.788675, 0.211325, 0.577350)),
        dot(pos,(FLOAT3)(0.577350, 0.577350, 0.577350))
    )*2;

    const long3 root = convert_long3(floor(pos));
    const FLOAT3 sub = pos - PASTE(convert_,FLOAT3)(root);
    FLOAT xyz[8];
    for(int dx=0;dx<2;dx++){
        for(int dy=0;dy<2;dy++){
            for(int dz=0;dz<2;dz++){
                xyz[dx<<2|dy<<1|dz]=cell_rand(seed, root + (long3)(dx,dy,dz));
            }
        }
    }
    FLOAT xy[4];
    xy[0b00] = mix(xyz[0b000],xyz[0b001],smoothstep(0,1,sub.z));
    xy[0b01] = mix(xyz[0b010],xyz[0b011],smoothstep(0,1,sub.z));
    xy[0b10] = mix(xyz[0b100],xyz[0b101],smoothstep(0,1,sub.z));
    xy[0b11] = mix(xyz[0b110],xyz[0b111],smoothstep(0,1,sub.z));

    FLOAT x[2];
    x[0b0] = mix(xy[0b00],xy[0b01],smoothstep(0,1,sub.y));
    x[0b1] = mix(xy[0b10],xy[0b11],smoothstep(0,1,sub.y));

    FLOAT n = mix(x[0],x[1],smoothstep(0,1,sub.x));

    return smoothstep(0,1,n);
}
