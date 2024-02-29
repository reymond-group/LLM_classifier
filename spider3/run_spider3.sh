#!/bin/bash
cd "$(dirname "$0")"

#######################################
# README!
#
# INPUT_LIST needs to be a file where each line contains the following fields
# <sequence name> <name with file path of fasta file>
# The file paths need to be either absolute or relative to this script.
#
# the NETWORK_DIR must contain each of the networks, as well as the pkl file 
# with normalisation values, split up in folders named i<iteration>-{ss,rest}.
#
#######################################
SECONDS=0

SAVE_DIR='./dbaasp_hemolysis/out/'
INPUT_LIST='./dbaasp_hemolysis_list'
NETWORK_DIR='./network_files/'

#######################################
source /home/markus/miniconda3/bin/activate spider3

#######################################
# Iteration 0
#######################################
ITER=0

###################
# SS
###################
echo "doing iteration ${ITER} - SS"
#SECONDS=0
python ./spider3_impute_np.py --saved_network_dir "${NETWORK_DIR}i${ITER}-ss" --input_file_list  ${INPUT_LIST} -o 'ss' 'ss8' -s ${SAVE_DIR} --save_ext ".i${ITER}s"
#echo ${SECONDS}

###################
# REST
###################
echo "doing iteration ${ITER} - ASA THETA TAU PHI PSI HSEa CN"
#SECONDS=0
python ./spider3_impute_np.py --saved_network_dir "${NETWORK_DIR}i${ITER}-rest" --input_file_list  ${INPUT_LIST} -o 'asa' 'ttpp' 'hsea' 'cn' -s ${SAVE_DIR} --save_ext ".i${ITER}r"
#echo ${SECONDS}


###################
# combine
###################
echo "combining both prediction files"
python ./script/combine_outputs_from_file_list.py ${INPUT_LIST} "${SAVE_DIR}" ".i${ITER}s" ".i${ITER}r" ".i${ITER}c" header="spd3 output - iteration ${ITER}"


###################
# convert to readable - optional.
###################
#cat ${INPUT_LIST} | awk '{ print $1 }' | xargs -I{} -P4 ./script/convert_raw_output_to_readable.py ${SAVE_DIR}{}.i${ITER}c ${SAVE_DIR}{}.i${ITER}


#######################################
# Iteration 1
#######################################
ITER=1

###################
# SS
###################
echo "doing iteration ${ITER} - SS"
python ./spider3_impute_np.py --saved_network_dir "${NETWORK_DIR}i${ITER}-ss" --input_file_list  ${INPUT_LIST} -o 'ss' 'ss8' -s ${SAVE_DIR} --save_ext ".i${ITER}s" --input_ext ".i$(( ITER - 1 ))c" --input_dir "${SAVE_DIR}"

###################
# REST
###################
echo "doing iteration ${ITER} - ASA THETA TAU PHI PSI HSEa CN"
python ./spider3_impute_np.py --saved_network_dir "${NETWORK_DIR}i${ITER}-rest" --input_file_list  ${INPUT_LIST} -o 'asa' 'ttpp' 'hsea' 'cn' -s ${SAVE_DIR} --save_ext ".i${ITER}r"  --input_ext ".i$(( ITER - 1 ))c" --input_dir "${SAVE_DIR}"


###################
# combine
###################
echo "combining both prediction files"
python ./script/combine_outputs_from_file_list.py ${INPUT_LIST} "${SAVE_DIR}" ".i${ITER}s" ".i${ITER}r" ".i${ITER}c" header="spd3 output - iteration ${ITER}"


###################
# convert to readable - optional.
###################
#cat ${INPUT_LIST} | awk '{ print $1 }' | xargs -I{} -P4 ./script/convert_raw_output_to_readable.py ${SAVE_DIR}{}.i${ITER}c ${SAVE_DIR}{}.i${ITER}
cat ${INPUT_LIST} | awk -v a=${SAVE_DIR} -v b=${ITER} '{ print a$1".i"b"c "$2" "a$1".i"b }' | xargs -I{} -P4 python script/convert_raw_output_to_readable_online.py {}

echo "Time taken - ${SECONDS} seconds"

