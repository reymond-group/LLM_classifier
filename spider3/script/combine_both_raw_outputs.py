#!/usr/bin/python

def combine_two_outputs(combined_name, ss_file, rest_file, header='combined raw file'):
  import numpy as np

  np.savetxt(combined_name, np.concatenate([np.loadtxt(ss_file), np.loadtxt(rest_file)], axis=1), header=header)
  
if __name__ == "__main__":
  import sys
  
  if len(sys.argv) == 4:
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]
    combine_two_outputs(output_file, file1, file2)
  elif len(sys.argv) == 5:
    header = sys.argv[4]
    combine_two_outputs(output_file, file1, file2, header=header)
  else:
    print 'invalid number of inputs'
  
  
