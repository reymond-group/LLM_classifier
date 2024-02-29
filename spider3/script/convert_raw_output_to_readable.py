#!/usr/bin/python

def convert_raw_file(input_filename, output_filename):
  import numpy as np

  raw_data = np.loadtxt(input_filename)
  

  ss_order = ['C','H','E']
  ss_ind = np.argmax(raw_data[:,0:3], axis=1)
  pred_ss = np.array([ss_order[i] for i in ss_ind])
  
  ss8_order = ["G","H","I","B","E","S","T","C","X"]
  ss8_ind = np.argmax(raw_data[:,3:11], axis=1)
  pred_ss8 = np.array([ss8_order[i] for i in ss8_ind])

  pred_asa = raw_data[:,11]

  raw_ttpp = raw_data[:,12:20] * 2 - 1;  
  pred_theta = np.rad2deg(np.arctan2(raw_ttpp[:,0], raw_ttpp[:,4]))
  pred_tau = np.rad2deg(np.arctan2(raw_ttpp[:,1], raw_ttpp[:,5]))
  pred_phi = np.rad2deg(np.arctan2(raw_ttpp[:,2], raw_ttpp[:,6]))
  pred_psi = np.rad2deg(np.arctan2(raw_ttpp[:,3], raw_ttpp[:,7]))

  pred_hsea_up = raw_data[:,20] * 50.
  pred_hsea_down = raw_data[:,21] * 65.

  pred_cn = raw_data[:,22] * 85.
  
  readable_data = np.zeros(pred_ss.size, 
          dtype=[('pred_ss', 'S1'), 
                 ('pred_ss8', 'S1'), 
                 ('pred_asa', float),
                 ('pred_phi', float),
                 ('pred_psi', float),
                 ('pred_theta', float),
                 ('pred_tau', float),
                 ('pred_hseau', float),
                 ('pred_hsead', float),
                 ('pred_cn', float) ])
                 
  readable_data['pred_ss'] = pred_ss
  readable_data['pred_ss8'] = pred_ss8
  readable_data['pred_asa'] = pred_asa
  readable_data['pred_phi'] = pred_phi
  readable_data['pred_psi'] = pred_psi
  readable_data['pred_theta'] = pred_theta
  readable_data['pred_tau'] = pred_tau
  readable_data['pred_hseau'] = pred_hsea_up
  readable_data['pred_hsead'] = pred_hsea_down
  readable_data['pred_cn'] = pred_cn
  
  np.savetxt(output_filename, readable_data, fmt="%s %s %4.3f %8.3f %8.3f %8.3f %8.3f %6.3f %6.3f %6.3f", header='SS SS8 ASA Phi Psi Theta Tau HSE_alpha_up HSE_alpha_down CN13')

  
if __name__ == "__main__":
  import sys
  
  if len(sys.argv) == 3:
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
  else:
    print 'invalid number of inputs'
    exit
  
  convert_raw_file(input_filename, output_filename)
  
  
