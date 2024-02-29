def combine_two_outputs(combined_name, ss_file, rest_file, header='spd3 output'):
  import numpy as np
  # get headers.
  with open(ss_file) as fp:
    h1 = fp.readline()
    # if the first char is # the line is the header.
    if h1[0] == '#':
      h1 = h1[1:-1]
    else:
      h1=None
  with open(rest_file) as fp:
    h2 = fp.readline()
    # if the first char is # the line is the header.
    if h2[0] == '#':
      h2 = h2[1:-1]
    else:
      h2=None
  
  if h1 != None or h2 != None:
    header = h1+h2
      
      
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
    
if __name__ == "__main__":
  import sys
    
  combined_outputs_from_file_list(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], header=sys.argv[6])
