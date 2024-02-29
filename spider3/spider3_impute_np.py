import sys
sys.path.append('./source')

import time
import random
import argparse
import os
import pickle
import numpy as np
import numpy.matlib
import scipy.io as sp
from scipy import special
import load_bioinf_data as load_data
import misc_functions as misc_functions



def run_RNN(inputs, matrix_in, bias, mask):
  "Run an RNN layer for a given input, matrix of weights and a bias"
  # Init
  temp_x = None
  x = inputs
  output = np.zeros([inputs.shape[0], inputs.shape[1], matrix_in.shape[1]/4])
  h = np.matrix(np.zeros((inputs.shape[0], matrix_in.shape[1]/4)))
  c = np.matrix(np.zeros((inputs.shape[0], matrix_in.shape[1]/4)))
  #Loop over inputs
  for iterator in range(inputs.shape[1]):
      temp_x = x[:,iterator,:]
      temp_x = np.concatenate((temp_x, h), axis=1)
      inprod = np.matmul(temp_x, matrix_in)
      finalprod = inprod + bias

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = np.split(finalprod, 4, axis=1)

      c = np.multiply(c, special.expit(f + 1.0)) + np.multiply(special.expit(i), np.tanh(j))
      h = np.multiply(np.tanh(c), special.expit(o))
      output[:,iterator,:] = h
  
  return (np.multiply(output, np.tile(mask[:,:,None], (1,1,matrix_in.shape[1]/4))), c)

def revseq(input, lengths):
    for i in range(input.shape[0]):
        input[i, 0:lengths[i], :] = input[i, (lengths[i]-1)::-1, :]
    return input

def softmax(x):
    # """
    # Compute softmax values for each sets of scores in x.
    
    # Rows are scores for each class. 
    # Columns are predictions (samples).
    # """
    # scoreMatExp = np.exp(np.asarray(x))
    # return scoreMatExp / scoreMatExp.sum(0)
    a = np.exp(x)
    b = np.sum(np.exp(x), axis = 1)
    b = np.tile(b[:,None], (1,x.shape[1]))
    c = np.divide(a, b)
    return c
    #return np.exp(x) / np.sum(np.exp(x), axis=1)

def num_batches(num_vec, batch_size):
  incomplete_batch = 1 if np.mod(num_vec, batch_size) else 0
  return num_vec/batch_size+incomplete_batch

def bioinf_output_nonlinearity(output_type, pred):
  # this function applies the nonlinear activation functions for the different output types.
  
  output_type = output_type.upper()
  if output_type == 'SS' or output_type == 'SS8':
    non_linear_output = softmax(pred)
  elif output_type == 'ASA' or output_type == 'HSEA' or output_type == 'HSEB' or output_type == 'CN' or output_type == 'CN13' or output_type == 'THETA' or output_type == 'TAU' or output_type == 'PHI' or output_type == 'PSI' or output_type == 'TT' or output_type == 'PP':
    non_linear_output = special.expit(pred)
  elif output_type == 'TTPP':
    non_linear_output = special.expit(pred)

  return non_linear_output

def brnn_impute(directory_to_saved_network, input_list, output_types, network_size=[[256, 256], [1024, 512]], directory_to_save_files=None, print_results=False, input_file_dir=None, save_file_ext='.spd3', input_file_ext='.spd3'):
    "This function is a wrapper for the main body of code in this file. It is called by any script options"

    # input_types is either the list of input types - same as brnn.py
    #   or it can be a filename - for example the casp data.
    #   if it is the filename we are currently making a heap of assumptions about what type of data it is.
    # output_types does the same thing. either the same as previously (for mat
    # files), a list of the outputs for casp, or <n_classes> for no accuracy
    # testing.

    scope_str = 'full'
    workdir = os.path
    if print_results is True:
        if not os.path.exists(directory_to_save_files):
            os.makedirs(directory_to_save_files)

    # Load data

    # Load training data normalisation stats
    fp = open(directory_to_saved_network +
            '/data_stats_' + scope_str + '.pkl', 'r')
    feat_mean, feat_var = pickle.load(fp)
    fp.close()

    test_seq_names, test_feat, feature_length = load_data.load_seqonly_input_wrapper(input_list, feat_mean, feat_var, input_file_dir=input_file_dir, input_file_ext=input_file_ext)

    true_label_ind, pred_label_ind, n_classes = load_data.get_outputs_list_stub(
        output_types)

    test_lengths = [len(tmp) for tmp in test_feat]

    # Network Parameters
    # n_input is the size of the features, i.e. 20 for PSSM
    n_input = feature_length

    # Load in data from pickled checkpoint that has already been converted
    # from TensorFlow
    checkpoint = open(directory_to_saved_network + '/network_best_full.pickle')
    # Two calls to pickle to load the two dumps in the file
    checkpoint_names = pickle.load(checkpoint)
    checkpoint_data = pickle.load(checkpoint)

#    output_weights = checkpoint_data[0]
#    output_bias = checkpoint_data[6]
#    FC0_weights = checkpoint_data[13]
#    FC0_bias = checkpoint_data[9]
#    FC1_weights = checkpoint_data[1]
#    FC1_bias = checkpoint_data[7]

#    RNN1_FW_matrix = checkpoint_data[10]
#    RNN1_BW_matrix = checkpoint_data[2]
#    RNN1_FW_bias = checkpoint_data[3]
#    RNN1_BW_bias = checkpoint_data[4]

#    RNN2_FW_matrix = checkpoint_data[12]
#    RNN2_BW_matrix = checkpoint_data[11]
#    RNN2_FW_bias = checkpoint_data[8]
#    RNN2_BW_bias = checkpoint_data[5]

    output_weights = checkpoint_data[13]
    output_bias = checkpoint_data[12]
    FC0_weights = checkpoint_data[9]
    FC0_bias = checkpoint_data[8]
    FC1_weights = checkpoint_data[11]
    FC1_bias = checkpoint_data[10]

    RNN1_FW_matrix = checkpoint_data[3]
    RNN1_BW_matrix = checkpoint_data[1]
    RNN1_FW_bias = checkpoint_data[2]
    RNN1_BW_bias = checkpoint_data[0]

    RNN2_FW_matrix = checkpoint_data[7]
    RNN2_BW_matrix = checkpoint_data[5]
    RNN2_FW_bias = checkpoint_data[6]
    RNN2_BW_bias = checkpoint_data[4]


    batch_size = 100
    input_feat = test_feat
    seq_len = test_lengths

    for i in xrange(0, num_batches(len(input_feat), batch_size)):
        #print "Doing batch ", i
        batch_ind = range(i*batch_size, np.minimum((i+1)*batch_size, len(input_feat)))
        batch_seq_lengths = [ seq_len[ind] for ind in batch_ind ]
        batch_max_length = max(batch_seq_lengths)
        batch_feat = np.array( [ np.concatenate((np.array(tmp), np.zeros((batch_max_length - tmp.shape[0], len(input_feat[0][0]))))) for tmp in [ input_feat[ind] for ind in batch_ind ] ] )
        batch_seq_len_mask = np.array( [ np.concatenate((np.ones(tmp), np.zeros(batch_max_length - tmp))) for tmp in batch_seq_lengths ] ) 
        batch_seq_names = [ test_seq_names[ind] for ind in batch_ind ]

        # Train/Test
        #Init and 0-pad
        x = batch_feat
        # Run Forwards RNN1
        RNN1_fw, RNN1_state_fw = run_RNN(x, RNN1_FW_matrix, RNN1_FW_bias, batch_seq_len_mask)
        # Run Back RNN1 (reverse input and then reverse output)
        RNN1_bw, RNN1_state_bw = run_RNN(revseq(x, batch_seq_lengths), RNN1_BW_matrix, RNN1_BW_bias, batch_seq_len_mask)
        RNN1_bw = revseq(RNN1_bw, batch_seq_lengths)
        # Combine RNN1
        RNN1_out = np.concatenate((RNN1_fw, RNN1_bw), axis=2)
        # Run Forwards RNN2
        RNN2_fw, RNN2_state_fw = run_RNN(RNN1_out, RNN2_FW_matrix, RNN2_FW_bias, batch_seq_len_mask)
        # Run Backwards RNN2
        RNN2_bw, RNN2_state_bw = run_RNN(revseq(RNN1_out, batch_seq_lengths), RNN2_BW_matrix, RNN2_BW_bias, batch_seq_len_mask)
        RNN2_bw = revseq(RNN2_bw, batch_seq_lengths)
        # Combine RNN2
        RNN2_out = np.concatenate((RNN2_fw, RNN2_bw), axis=2)
        FC_in = RNN2_out[0, 0:batch_seq_lengths[0], :]
        for i in range(1,RNN2_out.shape[0]):
            FC_in = np.concatenate((FC_in, RNN2_out[i, 0:batch_seq_lengths[i], :]))

        # FC Layer 1
        FC0 = np.matmul(FC_in, FC0_weights)
        FC0 = np.add(FC0, FC0_bias)
        FC0_out = np.maximum(FC0, 0)
        # FC Layer 2
        FC1 = np.matmul(FC0_out, FC1_weights)
        FC1 = np.add(FC1, FC1_bias)
        FC1_out = np.maximum(FC1, 0)
        # Output Layer
        OUT = np.matmul(FC1_out, output_weights)
        OUT_out = np.add(OUT, output_bias)

        pred = OUT_out
        linear_output = pred
        
        output_index_pred = pred_label_ind

        temp_non_linear_output = []

        for ind, output_type in enumerate(output_types):
            temp_non_linear_output.append(bioinf_output_nonlinearity(output_type, linear_output[:, output_index_pred[ind][0]:output_index_pred[ind][1]]))
        
        non_linear_output = np.concatenate((temp_non_linear_output), axis=1)
        #non_linear_output = softmax(pred)
        #We're done calculations. Save the results.
        if not os.path.exists(directory_to_save_files):
            print "Making directory " + directory_to_save_files
            os.makedirs(directory_to_save_files)
        misc_functions.save_predictions_to_file(non_linear_output, batch_seq_names, batch_seq_lengths, save_dir=directory_to_save_files, file_ext=save_file_ext, header='%s' % ', '.join(map(str, output_types)))

if __name__ == "__main__":
    import cProfile, pstats, StringIO

    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_network_dir',
                        dest="directory_to_saved_networks",
                        help="directory where the network and normalisation values are saved")
    parser.add_argument('--input_file_list',
                        dest="input_file_list",
                        help="list of inputs. each line should contain <seq name> <pssm file> <hhm file>.")
    parser.add_argument('-o', '--output_types',
                        nargs='+',
                        dest="output_types",
                        default=['ss'],
                        help='output types for the network.')
    parser.add_argument('-s', '--output_save_directory',
                        dest="directory_to_save_outputs",
                        default='./',
                        help='directory to save all files to.')
    parser.add_argument('--save_ext',
                        dest="save_ext",
                        default=".spd3",
                        help="file extension for the output files")
    parser.add_argument('--input_ext',
                        dest="input_ext",
                        help="file extension of the previous outputs (being used as inputs for this iteration)",
                        default=None)
    parser.add_argument('--input_dir',
                        dest="input_dir",
                        help="directory of input files",
                        default=None)

    args = parser.parse_args()


#    pr = cProfile.Profile()
#    pr.enable()
    
    brnn_impute(args.directory_to_saved_networks, args.input_file_list, args.output_types, print_results=True,
                directory_to_save_files=args.directory_to_save_outputs, save_file_ext=args.save_ext, input_file_dir=args.input_dir, input_file_ext=args.input_ext)
                
#    s = StringIO.StringIO()
#    sortby = 'cumulative'
#    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#    ps.print_stats()
#    print s.getvalue()
