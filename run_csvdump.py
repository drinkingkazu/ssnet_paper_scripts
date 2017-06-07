import os,sys
DATA_DIR=os.getcwd()
for argv in sys.argv:
    if argv.startswith('datadir='):
        DATA_DIR=argv.replace('datadir=','')
files=['%s/%s' % (DATA_DIR,f) for f in os.listdir(DATA_DIR) if f.startswith(sys.argv[1]) and f.endswith('.caffemodel.h5')]
for f in files: 
    cmd = 'python %s/csvdump.py %s' % (DATA_DIR,f)
    if len(sys.argv)>2:
        for x in xrange(len(sys.argv)-2):
            cmd += ' %s ' % sys.argv[2+x]
    os.system(cmd)
