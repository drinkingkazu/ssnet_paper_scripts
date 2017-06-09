#!/bin/bash

LOCKFILE=/tmp/gpu$1_lock.txt
if [ -e ${LOCKFILE} ] && kill -0 `cat ${LOCKFILE}`; then
    exit
fi

# make sure the lockfile is removed when we exit and then claim it                                                                                                                
trap "rm -f ${LOCKFILE}; exit" INT TERM EXIT
echo $$ > ${LOCKFILE}

source /home/drinkingkazu/sw/larcv/configure.sh
source /home/drinkingkazu/sw/caffe_devel/configure.sh
python /data/drinkingkazu/UBDeconvNet/dlmc_mcc8_multipvtx_v01/run_csvdump.py $1 $2 gpumem=10000
 
