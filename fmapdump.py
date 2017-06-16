import os, sys

DATA_DIR= os.getcwd()
if not sys.argv[1].find('caffemodel') >=0:
    sys.stderr.write('caffemodel file must be the 1st argument\n')
    sys.exit(1)

AVAILABLE_FLAVOUR = {'val'  : '/stage2/drinkingkazu/dl_production_v01/dlmc_mcc8_multipvtx_v01/val.root',
                     '1e1p' : '/stage2/drinkingkazu/dl_production_v01/dlmc_mcc8_multipvtx_v01/val_1e1p.root',
                     'nue'  : '/stage2/drinkingkazu/dl_production_v01/dlmc_mcc8_multipvtx_v01/val_nue.root',
                     'numu' : '/stage2/drinkingkazu/dl_production_v01/dlmc_mcc8_multipvtx_v01/val_numu.root',
                     '1e1p_lowE' : '/stage2/drinkingkazu/dl_production_v01/dlmc_mcc8_multipvtx_v01/val_1e1p_lowE.root'
                     }

MODEL        = sys.argv[1]
PROTO        = 'ures'
GPUMEM       = 10000
TARGET_ENTRY = -1
PLANE        = 2
FLAVOUR      = None

for argv in sys.argv:
    if argv.startswith('datadir='):
        DATA_DIR = argv.replace('datadir=','')
    if argv.startswith('event='):
        TARGET_ENTRY = int(argv.replace('event=',''))
    if argv.startswith('gpumem='):
        GPUMEM = int(argv.replace('gpumem=',''))
    if argv.startswith('flavour='):
        FLAVOUR = argv.replace('flavour=','')
    if argv.startswith('plane='):
        PLANE = int(argv.replace('plane=',''))
    if argv.startswith('proto='):
        PROTO = argv.replace('proto=','')

CSVNAME = MODEL.replace('.caffemodel.h5','_%s_plane%d.csv' % (FLAVOUR,PLANE))
if PLANE >2 or PLANE<0:
    print 'plane=%s' % str(PLANE),'not supported!'
    sys.exit(1)
if not FLAVOUR in AVAILABLE_FLAVOUR.keys():
    print 'flavour=%s' % str(FLAVOUR),'not supported!'
    sys.exit(1)
if CSVNAME == MODEL:
    print 'Could not rename MODEL to CSV ...'
    sys.exit(1)
if os.path.isfile(CSVNAME):
    print 'Already processed:',CSVNAME
    sys.exit(1)

#
# Generate temp file 
#
import tempfile, filler_template
filler_cfg=tempfile.NamedTemporaryFile()
filler_cfg.write(filler_template.get_template().replace('FILEPATH',AVAILABLE_FLAVOUR[FLAVOUR]).replace('PLANE',str(PLANE)))
filler_cfg.flush()

exec('from %s_template import get_template as proto_template' % PROTO)
proto_cfg=tempfile.NamedTemporaryFile()
proto_cfg.write(proto_template().replace('CONFIG_NAME',filler_cfg.name))
proto_cfg.flush()

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

net = caffe.Net(proto_cfg.name, MODEL, caffe.TEST)
filler = larcv.ThreadFillerFactory.get_filler("DataFiller")
num_events = filler.get_n_entries()
if TARGET_ENTRY < 0 or TARGET_ENTRY >= num_events:
    print "TARGET_ENTRY",TARGET_ENTRY,"is not valid! (entries in file",num_events
    sys.exit(1)

print
print 'Processing',MODEL
print 'Entry',TARGET_ENTRY
print

npx_sum={1:0,2:0}
npx_sum_correct={1:0,2:0}

filler.set_next_index(TARGET_ENTRY)

net.forward()

while filler.thread_running():
    time.sleep(0.001)

events = filler.processed_events()
entries = filler.processed_entries()

adcimgs = net.blobs["segdata"].data
labels  = net.blobs["seglabel"].data
scores  = net.blobs["softmax"].data

total_npx = 0
total_correct_npx = 0

adcimg = adcimgs[0] # ADC raw image
label  = labels[0]  # Segmentation label raw image
score  = scores[0].argmax(axis=0) # Resulting type image
score  = score.reshape([ adcimg.shape[0] ] + list(score.shape) ) # reshape

ana_pixel_map = (adcimg > 20) * (label > 0.5)

for t in [1,2]:
    #print '...',
    type_pixel_map = (label == t) * ana_pixel_map
    type_npx = type_pixel_map.sum()
    type_npx_correct = (label[type_pixel_map] == score[type_pixel_map]).sum()
    type_npx_res = (score[ana_pixel_map] == t).sum()

    total_npx += type_npx
    total_correct_npx += type_npx_correct
    
    npx_sum[t] += type_npx
    npx_sum_correct[t] += type_npx_correct

    #print '...', int(float(total_correct_npx)/float(total_npx) * 100.)/100.
    
msg = 'Entry %4d ... Total npx: %6d ... Shower: %6d / %6d (%.2f%%) ... Track: %6d / %6d (%.2f%%)'
temp_frac1 = 0.
if npx_sum[1]: temp_frac1 = int(float(npx_sum_correct[1]) / npx_sum[1] * 10000.) / 100.
temp_frac2 = 0.
if npx_sum[2]: temp_frac2 = int(float(npx_sum_correct[2]) / npx_sum[2] * 10000.) / 100.
msg = msg % (TARGET_ENTRY,
             npx_sum[1] + npx_sum[2],
             npx_sum_correct[1], npx_sum[1], temp_frac1,
             npx_sum_correct[2], npx_sum[2], temp_frac2)
print msg

fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
fmappng=plt.imshow(adcimg[0],cmap='jet',interpolation='none')
fmappng.write_png('Event%04d_data.png' % TARGET_ENTRY)
plt.close(fig)

fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
fmappng=plt.imshow(label[0] * 80.,vmin=0,vmax=255,cmap='jet',interpolation='none')
fmappng.write_png('Event%04d_label.png' % TARGET_ENTRY)
plt.close(fig)

fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
fmappng=plt.imshow(score[0] * 80.,vmin=0,vmax=255,cmap='jet',interpolation='none')
fmappng.write_png('Event%04d_ssnet.png' % TARGET_ENTRY)
plt.close(fig)

fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
fmappng=plt.imshow(scores[0][0],vmin=0,vmax=1.,cmap='jet',interpolation='none')
fmappng.write_png('Event%04d_softmax_Type0.png' % TARGET_ENTRY)
plt.close(fig)

fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
fmappng=plt.imshow(scores[0][1],vmin=0,vmax=1.,cmap='jet',interpolation='none')
fmappng.write_png('Event%04d_softmax_Type1.png' % TARGET_ENTRY)
plt.close(fig)

fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
fmappng=plt.imshow(scores[0][2],vmin=0,vmax=1.,cmap='jet',interpolation='none')
fmappng.write_png('Event%04d_softmax_Type2.png' % TARGET_ENTRY)
plt.close(fig)



def sumpool(ar,n,m):
    n=int(n)
    m=int(m)
    print 'requested sum pool from array with shape =',ar.shape,'by a factor of (%d,%d)' % (n,m)
    if not len(ar.shape) == 2:
        print 'Only use 2D array!'
        return
    if ar.shape[0]%n or ar.shape[1]%m:
        print 'Shape',ar.shape,'cannot be contracted by (%d,%d)' % (n,m)
    res=np.zeros(shape=(ar.shape[0]/n,ar.shape[1]/m))
    for i in xrange(ar.shape[0]):
        for j in xrange(ar.shape[1]):
            res[i/n][j/m] += ar[i][j]
    print 'returning pooled map with shape =',res.shape
    return res

# loop over types
#layer_names=['conv0','res1a','res1b','res2a','res2b','res3a','res3b','res4a','res4b','res5a','res5b',
#             'deconv0_deconv','res6a','res6b','deconv1_deconv','res7a','res7b','deconv2_deconv','res8a','res8b',
#             'deconv3_deconv','res9a','res9b','deconv4_deconv','conv10']
#layer_names=['deconv0_deconv','res6a','res6b','deconv1_deconv','res7a','res7b','deconv2_deconv','res8a','res8b',
#             'deconv3_deconv','res9a','res9b','deconv4_deconv','conv10']
layer_names=[]

if SAVE_ALL:
    # loop over layers
    for name_idx in xrange(len(layer_names)):
        name = layer_names[name_idx]
        fmap_v = net.blobs[name].data[0]
        for idx in xrange(len(fmap_v)):
            fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
            fmappng=plt.imshow(fmap_v[idx],cmap='jet',interpolation='none')
            fmappng.write_png('%s_plane%d_Layer%02d_%s_%04d.png' % (PROTO,PLANE,name_idx,name,idx))
            plt.close(fig)

for t in [1,2]:
    label_map = {}
    label_map[(1,1)] = ((label == t).astype(np.int32))[0]
    # loop over layers
    for name_idx in xrange(len(layer_names)):
        name = layer_names[name_idx]
        # finding max response map for given type
        fmap_v = net.blobs[name].data[0]
        fmap_index = 0
        fmap_maxsum = 0

        if len(fmap_v) <1 : continue

        # generate index map
        n = label[0].shape[0]/fmap_v[0].shape[0]
        m = label[0].shape[1]/fmap_v[0].shape[1]
        label_key = (n,m)
        if not label_key in label_map:
            label_map[label_key] = sumpool(label_map[(1,1)],n,m)
            label_map[label_key] = (label_map[label_key] > 0).astype(np.int32)

        factor_signal = label_map[label_key]

        for idx in xrange(len(fmap_v)):
            fmap_signal = fmap_v[idx] * factor_signal
            if fmap_signal.sum() > fmap_maxsum:
                fmap_maxsum = fmap_signal.sum()
                fmap_index = idx
        fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
        fmappng=plt.imshow(fmap_v[fmap_index],cmap='jet')
        fmappng.write_png('Type%d_%s_plane%d_Layer%02d_%s_%04d.png' % (t,PROTO,PLANE,name_idx,name,fmap_index))
        plt.close(fig)

