import os, sys

DATA_DIR= os.getcwd()
if not sys.argv[1].find('caffemodel') >=0:
    sys.stderr.write('caffemodel file must be the 1st argument\n')
    sys.exit(1)

MODEL        = sys.argv[1]
GPUMEM       = 10000
STOP_COUNTER = -1
for argv in sys.argv:
    if argv.startswith('datadir='):
        DATA_DIR = argv.replace('datadir=','')
    if argv.startswith('nevents='):
        STOP_COUNTER = int(argv.replace('nevents=',''))
    if argv.startswith('gpumem='):
        GPUMEM = int(argv.replace('gpumem=',''))

PROTO   = "%s/ana.prototxt" % DATA_DIR
CSVNAME = MODEL.replace('.caffemodel.h5','.csv')

if CSVNAME == MODEL:
    print 'Could not rename MODEL to CSV ...'
    sys.exit(1)
if os.path.isfile(CSVNAME):
    print 'Already processed:',CSVNAME
    sys.exit(1)
if not os.path.isfile(PROTO):
    print 'PROTO not found:',PROTO
    sys.exit(1)

from choose_gpu import pick_gpu
GPUID=pick_gpu(GPUMEM,caffe_gpuid=True)
if GPUID < 0:
    sys.stderr.write('No available GPU with memory %d\n' % GPUMEM)
    sys.exit(1)

import matplotlib 
matplotlib.use('Agg')
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import ROOT as rt
import time
from ROOT import larcv
import matplotlib.pyplot as plt
import caffe
caffe.set_mode_gpu()
caffe.set_device(GPUID)

BATCH_CTR=5
net = caffe.Net( PROTO, MODEL, caffe.TEST)
filler = larcv.ThreadFillerFactory.get_filler("DataFiller")
num_events = filler.get_n_entries()
if STOP_COUNTER <0:
    STOP_COUNTER = num_events

print
print 'Processing',MODEL
print 'Total number of events:',num_events
print 'Stop after processing:',STOP_COUNTER
print 'Batch size:', BATCH_CTR
print

event_counter = 0
current_index = 0
last_index = 0

npx_sum={1:0, 2:0}
npx_sum_correct={1:0, 2:0}

filler.set_next_index(current_index)
fout=open(CSVNAME,'w')
fout.write('entry,label_shower_npx,reco_shower_npx,correct_shower_npx,label_track_npx,reco_track_npx,correct_track_npx,total_npx,correct_npx\n')
for ibatch in range(0,num_events / BATCH_CTR+1):

    last_index = current_index    

    current_index = ibatch * BATCH_CTR

    #print 'current_index',current_index
    #filler.set_next_index(current_index)

    #print 'Batch:',ibatch

    num_entries = num_events - current_index

    if num_entries > BATCH_CTR: num_entries = BATCH_CTR

    net.forward()

    while filler.thread_running():
        time.sleep(0.001)

    events = filler.processed_events()
    entries = filler.processed_entries()

    if events.size() != BATCH_CTR:
        print "Batch counter mis-match!"
        raise Exception

    adcimgs = net.blobs["segdata"].data
    labels  = net.blobs["seglabel"].data
    scores  = net.blobs["softmax"].data

    for index in xrange(len(scores)):

        line ='%d' % entries[index]

        total_npx = 0
        total_correct_npx = 0

        adcimg = adcimgs[index] # ADC raw image
        label  = labels[index]  # Segmentation label raw image
        score  = scores[index].argmax(axis=0) # Resulting type image
        score  = score.reshape([ adcimg.shape[0] ] + list(score.shape) ) # reshape

        ana_pixel_map = (adcimg > 20) * (label > 0.5)

        for t in [1,2]:
            #print '...',
            type_pixel_map = (label == t) * ana_pixel_map
            type_npx = type_pixel_map.sum()
            type_npx_correct = (label[type_pixel_map] == score[type_pixel_map]).sum()
            type_npx_res = (score[ana_pixel_map] == t).sum()

            line +=',%d.,%d.,%d.' % (type_npx,type_npx_res,type_npx_correct)
            #if type_npx:
            #    print int(float(type_npx_res)/float(type_npx) * 100.) / 100.,
            #else:
            #    print '  0.00',
            #if type_npx:
            #    print int(float(type_npx_correct)/float(type_npx) * 100.) / 100.,
            #else:
            #    print '  0.00',

            total_npx += type_npx
            total_correct_npx += type_npx_correct

            npx_sum[t] += type_npx
            npx_sum_correct[t] += type_npx_correct

        #print '...', int(float(total_correct_npx)/float(total_npx) * 100.)/100.

        line += ',%d.,%d.\n' % (total_npx,total_correct_npx)
        fout.write(line)

        msg = 'Entry %4d ... Total npx: %6d ... Shower: %6d / %6d (%.2f%%) ... Track: %6d / %6d (%.2f%%)\r'
        temp_frac1 = 0.
        if npx_sum[1]: temp_frac1 = int(float(npx_sum_correct[1]) / npx_sum[1] * 10000.) / 100.
        temp_frac2 = 0.
        if npx_sum[2]: temp_frac2 = int(float(npx_sum_correct[2]) / npx_sum[2] * 10000.) / 100.
        msg = msg % (event_counter,
                     npx_sum[1] + npx_sum[2],
                     npx_sum_correct[1], npx_sum[1], temp_frac1,
                     npx_sum_correct[2], npx_sum[2], temp_frac2)
        sys.stdout.write(msg)
        sys.stdout.flush()
        if (event_counter+1) % 100 == 0: 
            print

        event_counter += 1
        if event_counter >= STOP_COUNTER:
            break

    if num_entries < BATCH_CTR:
        break

    if event_counter >= STOP_COUNTER:
        break
fout.close()

