#!/home/tranus/pyenv/bin/python

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import pyopencl as cl
import multiprocess as mp
import random
from time import sleep
import PIL
from sys import getsizeof as sizeof
from math import *
import os 
import contextlib

this_file_path = os.path.dirname(os.path.realpath(__file__))
print(hex(random.getrandbits(64)))

#in pixels (AKA simulation units)
#affects the size of the area simulated, not the accuracy
IMG_SIZE=1024
#maximum, in sim units
PERIOD=1024
#modifier, unitless
SLOPE=1
SIM_STEPS=10000
SIM_BATCH=100
#modifier, lower increases stability
SIM_RATE=0.25

#in sim units
VERT_RANGE= SLOPE*PERIOD/8
print("VERT_RANGE: {:.0f}".format(VERT_RANGE))
SEED=np.uint64(random.getrandbits(64))

BUF_SHAPE=(3,IMG_SIZE,IMG_SIZE)
BUF_LAND=0
BUF_WATER=1
BUF_SEDIMENT=2

FLOAT='float'
NPFLOAT=np.float32

mgr = mp.Manager()
draw_queue = mgr.Queue()
should_close=mp.Event()
stop_drawing=mp.Event()
draw_ready = mp.Event()

def draw():

    matplotlib.use('GTK3Agg')
    plt.ion()
    fig = plt.figure(figsize=(12,12), num='Terrain')
    plt.axes((0,0,1,1))
    def on_close(*_):
        global should_close
        should_close.set()
    fig.canvas.mpl_connect('close_event',on_close)
    plt.axis('off')

    water_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('water_cmap',[(0,0,1,0),(0,0,1,0.5)])

    land_img_ax = plt.imshow(np.zeros((IMG_SIZE,IMG_SIZE),dtype=NPFLOAT), cmap='gray', vmin=0, vmax=VERT_RANGE)
    water_img_ax = plt.imshow(np.zeros((IMG_SIZE,IMG_SIZE),dtype=NPFLOAT), cmap=water_cmap, vmin=0, vmax=1)

    while(not stop_drawing.is_set()):
        if(not draw_queue.empty()):
            img=draw_queue.get()
            land_img_ax.set_data(img[BUF_LAND])
            water_img_ax.set_data(img[BUF_WATER])
        else:
            draw_ready.set()
            fig.canvas.flush_events()
            fig.canvas.draw_idle()
            sleep(0.016)

draw_proc=mp.Process(target=draw)
draw_proc.daemon=True
draw_proc.start()

ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

start=timer()
with open('main.cl','r') as maincl:
    prg = cl.Program(ctx, str(maincl.read())).build(options=[f'-I{this_file_path} -DFLOAT={FLOAT} -cl-std=CL2.0'])
krn={}
for kernel in prg.all_kernels():
    krn[kernel.function_name]=kernel
print('kernel compilation: {:0.4f}s'.format(timer()-start))

host_buf = np.zeros(BUF_SHAPE,dtype=NPFLOAT)
tmp_buf = np.zeros((IMG_SIZE,IMG_SIZE),dtype=NPFLOAT)
current_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_buf)
oro_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=tmp_buf)
del tmp_buf
next_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_buf)

krn['supply_buffers'](queue, (1,1), (1,1), current_buf, next_buf, np.int64(IMG_SIZE), np.int64(IMG_SIZE))

start=timer()
krn['init_height'](queue, (IMG_SIZE,IMG_SIZE), None, SEED, NPFLOAT(1.0/PERIOD), NPFLOAT(0), NPFLOAT(VERT_RANGE))
queue.finish()
print("noisegen {:0.4f}s".format(timer()-start))

total_step_time = 0
total_steps = 0
very_start=timer()
for batch in range(SIM_STEPS//SIM_BATCH):
    if(should_close.is_set()):
        break

    for b in range(SIM_BATCH):
        step = batch*SIM_BATCH + b
        #krn['orogeny'](queue, (IMG_SIZE,IMG_SIZE), None, NPFLOAT(step*SIM_RATE/10), NPFLOAT(1.0/PERIOD), NPFLOAT(1), NPFLOAT(SIM_RATE))
        #krn['copy_back'](queue, (IMG_SIZE,IMG_SIZE), None)
        krn['gravity'](queue, (IMG_SIZE,IMG_SIZE), None, NPFLOAT(1), NPFLOAT(SIM_RATE))
        krn['copy_back'](queue, (IMG_SIZE,IMG_SIZE), None)
        krn['hydro_precip'](queue, (IMG_SIZE,IMG_SIZE), None, NPFLOAT(0.001*SIM_RATE))
        krn['copy_back'](queue, (IMG_SIZE,IMG_SIZE), None)
        for p in range(9):
            krn['hydro_flow_part'](queue, ((IMG_SIZE+2)//3,(IMG_SIZE+2)//3), None, np.int8(p), NPFLOAT(SIM_RATE))
        krn['copy_back'](queue, (IMG_SIZE,IMG_SIZE), None)
        krn['hydro_sink'](queue, (IMG_SIZE,IMG_SIZE), None, NPFLOAT(1))
        krn['copy_back'](queue, (IMG_SIZE,IMG_SIZE), None)
    
    start=timer()    
    queue.finish()
    elapsed=timer()-start

    total_step_time+=elapsed
    total_steps+=SIM_BATCH
    print('\r',' '*50,end='')
    print('\rstep {}\t{:0.4f}s'.format(batch*SIM_BATCH,elapsed/SIM_BATCH),end='')

    #send image to be drawn if the drawer is ready, or if on the last step
    if(draw_ready.is_set()):
        cl.enqueue_copy(queue,host_buf,current_buf)
        draw_queue.put(host_buf)
        draw_ready.clear()

cl.enqueue_copy(queue, host_buf, current_buf)
draw_queue.put(host_buf)

print(np.min(host_buf[0]))
print(np.max(host_buf[0]))

total_elapsed = timer()-very_start
if(SIM_STEPS!=0):
    print('\navg step time: {:0.4f}s'.format(total_step_time/total_steps))
    print('total time: {:0.4f}s'.format(total_elapsed))
    print('overhead: {:0.4f}s'.format(total_elapsed-total_step_time))

start=timer()
imdata = np.zeros((IMG_SIZE,IMG_SIZE),dtype=np.int32)
imbuf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=imdata)
krn['to_png'](queue, (IMG_SIZE, IMG_SIZE), None, current_buf, imbuf, np.int64(0), NPFLOAT(0), NPFLOAT(VERT_RANGE))
cl.enqueue_copy(queue, imdata, imbuf)
img = PIL.Image.fromarray(imdata,'I')
img.save('out.png')
print(f'out.png saved: {timer()-start:0.4f}s')


while(not should_close.is_set()):
    sleep(0.1)

stop_drawing.set()
draw_proc.join()