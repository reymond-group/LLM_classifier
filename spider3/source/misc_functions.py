import numpy as np

def num_batches(num_vec, batch_size):
  incomplete_batch = 1 if np.mod(num_vec, batch_size) else 0
  return num_vec/batch_size+incomplete_batch
  
  
def save_predictions_to_file(network_output, seq_names, seq_lengths, save_dir='.', file_ext='.spd3', header=''):
  # this function should take all of the network's predictions and save them to
  # individual files (in the save_dir directory, using file_ext as the file
  # extension).
  # header will be printed as a header to the file. This may be (for example)
  # the output types.
  
  # split network_output (which will be a large numpy array) into a list.
  # each element of the list should be a single sequence.
  temp_seq_lengths = [0,]+seq_lengths
  network_output_list = []
  for ind in range(len(seq_lengths)):
    network_output_list.append(network_output[sum(temp_seq_lengths[0:ind+1]):sum(temp_seq_lengths[0:ind+2]),:])
    
  # save each of those sequence predictions to a file.
  for ind, pred in enumerate(network_output_list):
    # write the prediction to a file
    str_name = save_dir+'/'+seq_names[ind]+file_ext
    np.savetxt(str_name, pred, header=header)
      
def combine_two_outputs(combined_name, ss_file, rest_file, header='spd3 output'):
  np.savetxt(combined_name, np.concatenate([np.loadtxt(ss_file), np.loadtxt(rest_file)], axis=1), header=header)
  
def combined_outputs_from_file_list(file_list, output_file_dir, ext_1, ext_2, combined_ext, header='spd3 output'):
  # file_list is the same as the file list input to the brnn_impute function,
  # ie - seq_name, seq_pssm_file, seq_hmm_file
  
  with open(file_list) as fp:
    lines = fp.readlines()
  
  for line in lines:
    
    temp_line = line.split()
    seq_name = temp_line[0]
    
    combine_two_outputs(output_file_dir + seq_name + combined_ext, output_file_dir + seq_name + ext_1, output_file_dir + seq_name + ext_2, header=header)
