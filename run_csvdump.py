import os,sys
DATA_DIR='/data/drinkingkazu/UBDeconvNet/dlmc_mcc8_multipvtx_v01'
files=['%s/%s' % (DATA_DIR,f) for f in os.listdir(DATA_DIR) if f.startswith(sys.argv[1]) and f.endswith('.caffemodel.h5')]
for f in files: 
    cmd = 'python %s/csvdump.py %s' % (DATA_DIR,f)
    if len(sys.argv)>2:
        for x in xrange(len(sys.argv)-2):
            cmd += ' %s ' % sys.argv[2+x]
    cmd += ' datadir=%s ' % DATA_DIR
    os.system(cmd)
