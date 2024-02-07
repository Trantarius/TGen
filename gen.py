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

IMG_SIZE=1024
SIM_STEPS=500
BUF_SHAPE=IMG_SIZE,IMG_SIZE

FLOAT='float'
NPFLOAT=np.float32

(draw_pipe,gen_pipe)=mp.Pipe()
should_close=mp.Event()
stop_drawing=mp.Event()

def draw():

    matplotlib.use('GTK3Agg')
    plt.ion()
    fig = plt.figure(figsize=(12,12))
    plt.axes((0,0,1,1))
    def on_close(*_):
        global should_close
        should_close.set()
    fig.canvas.mpl_connect('close_event',on_close)
    plt.axis('off')
    img_ax = plt.imshow(np.zeros((IMG_SIZE,IMG_SIZE),dtype=NPFLOAT), cmap='gray', vmin=0, vmax=100)

    while(not stop_drawing.is_set()):
        if(gen_pipe.poll()):
            img=gen_pipe.recv()
            img_ax.set_data(img)
        else:
            fig.canvas.flush_events()
            fig.canvas.draw_idle()
    gen_pipe.close()

draw_proc=mp.Process(target=draw)
draw_proc.start()

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

with open('main.cl','r') as maincl:
    prg = cl.Program(ctx, str(maincl.read())).build(options=[f'-DFLOAT={FLOAT}'])
krn={}
for kernel in prg.all_kernels():
    krn[kernel.function_name]=kernel

host_buf = np.zeros((IMG_SIZE,IMG_SIZE),dtype=NPFLOAT)
current_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_buf)
next_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_buf)
def swap_buffers():
    global current_buf
    global next_buf
    tmp=current_buf
    current_buf=next_buf
    next_buf=tmp

start=timer()
krn['fractal_warp_noisefill'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, NPFLOAT(random.random()*(1<<10)), NPFLOAT(1), NPFLOAT(1))
cl.enqueue_copy(queue,host_buf,current_buf)
print("noisegen ",timer()-start)
krn['map_range'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, 
    NPFLOAT(np.min(host_buf)), NPFLOAT(np.max(host_buf)), NPFLOAT(0), NPFLOAT(100))
queue.flush()

blur_conv = np.array([
    [0,0,1,0,0],
    [0,1,2,1,0],
    [1,2,4,2,1],
    [0,1,2,1,0],
    [0,0,1,0,0]
],dtype=NPFLOAT)
blur_conv /= np.sum(blur_conv)
blur_conv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=blur_conv)

for step in range(SIM_STEPS):
    print('step ',step)

    if(should_close.is_set()):
        break

    krn['flatten_slopes'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, next_buf, NPFLOAT(0.25))
    
    queue.flush()
    swap_buffers()

    #send image to be drawn if the drawer is ready, or if on the last step
    if(not gen_pipe.poll() or step==SIM_STEPS-1):
        cl.enqueue_copy(queue,host_buf,current_buf)
        draw_pipe.send(host_buf)

#krn['map_range'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, 
#    NPFLOAT(np.min(host_buf)), NPFLOAT(np.max(host_buf)), NPFLOAT(0), NPFLOAT(1))
#cl.enqueue_copy(queue,host_buf,current_buf)

imdata = np.zeros((IMG_SIZE,IMG_SIZE),dtype=np.int32)
imbuf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=imdata)
krn['map_range'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, NPFLOAT(0), NPFLOAT(100), NPFLOAT(0), NPFLOAT(1))
krn['to_int32'](queue, (IMG_SIZE,IMG_SIZE), None, current_buf, imbuf)
cl.enqueue_copy(queue, imdata, imbuf)

img = PIL.Image.fromarray(imdata,'I')
img.save('out.png')


while(not should_close.is_set()):
    sleep(0.1)

stop_drawing.set()
draw_proc.join()