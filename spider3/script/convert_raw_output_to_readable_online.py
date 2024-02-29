def convert_rasa_to_asa(rasa, aa):
  rnam1_std0 = "ACDEFGHIKLMNPQRSTVWYX"
  ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
         185, 160, 145, 180, 225, 115, 140, 155, 255, 230)
  asa_conversion_dict = {'A':115,'C':135,'D':150,'E':190,'F':210,
                         'G':75,'H':195,'I':175,'K':200,'L':170,
                         'M':185,'N':160,'P':145,'Q':180,'R':225,
                         'S':115,'T':140,'V':155,'W':255,'Y':230}
 
  asa = []
  for i in range(len(aa)):
    asa.append(rasa[i] * asa_conversion_dict[aa[i]])
  return(asa)

def convert_raw_file(input_filename, seq_filename, output_filename):
  import numpy as np

  with open(seq_filename, 'r') as fp:
    fp.readline()
    seq=fp.readline()
    
  aa_list = []
  for aa in seq.strip():
    aa_list.append(aa)
    
    
  raw_data = np.loadtxt(input_filename)  

  ss_order = ['C','H','E']
  ss_ind = np.argmax(raw_data[:,0:3], axis=1)
  pred_ss = [ss_order[i] for i in ss_ind]
  
  ss8_order = ["G","H","I","B","E","S","T","C","X"]
  ss8_ind = np.argmax(raw_data[:,3:11], axis=1)
  pred_ss8 = [ss8_order[i] for i in ss8_ind]

  pred_rasa = raw_data[:,11]
  pred_asa = np.array(convert_rasa_to_asa(pred_rasa,aa_list))

  raw_ttpp = raw_data[:,12:20] * 2 - 1;  
  pred_theta = np.rad2deg(np.arctan2(raw_ttpp[:,0], raw_ttpp[:,4]))
  pred_tau = np.rad2deg(np.arctan2(raw_ttpp[:,1], raw_ttpp[:,5]))
  pred_phi = np.rad2deg(np.arctan2(raw_ttpp[:,2], raw_ttpp[:,6]))
  pred_psi = np.rad2deg(np.arctan2(raw_ttpp[:,3], raw_ttpp[:,7]))

  pred_hsea_up = raw_data[:,20] * 50.
  pred_hsea_down = raw_data[:,21] * 65.

  pred_cn = raw_data[:,22] * 85.
  
  readable_data = np.zeros(len(pred_ss), 
          dtype=[('#', int), 
                 ('AA', 'U1'), 
                 ('pred_ss', 'U1'), 
                 ('pred_ss8', 'U1'), 
                 ('pred_asa', float),
                 ('pred_phi', float),
                 ('pred_psi', float),
                 ('pred_theta', float),
                 ('pred_tau', float),
                 ('pred_hseau', float),
                 ('pred_hsead', float),
                 ('pred_cn', float),
                 ('ss3_probc', float),
                 ('ss3_probe', float),
                 ('ss3_probh', float),
                 ('ss8_probg', float),
                 ('ss8_probh', float),
                 ('ss8_probi', float),
                 ('ss8_probb', float),
                 ('ss8_probe', float),
                 ('ss8_probs', float),
                 ('ss8_probt', float),
                 ('ss8_probc', float),
                 ])
                 
  readable_data['#'] = np.array(range(1,len(pred_ss)+1))
  readable_data['AA'] = np.array(aa_list)
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
#  readable_data['ss3_prob'] = np.array(raw_data[:,0], raw_data[:,2], raw_data[:,1])
  readable_data['ss3_probc'] = np.array(raw_data[:,0])
  readable_data['ss3_probe'] = np.array(raw_data[:,2])
  readable_data['ss3_probh'] = np.array(raw_data[:,1])
  readable_data['ss8_probg'] = np.array(raw_data[:,3])
  readable_data['ss8_probh'] = np.array(raw_data[:,4])
  readable_data['ss8_probi'] = np.array(raw_data[:,5])
  readable_data['ss8_probb'] = np.array(raw_data[:,6])
  readable_data['ss8_probe'] = np.array(raw_data[:,7])
  readable_data['ss8_probs'] = np.array(raw_data[:,8])
  readable_data['ss8_probt'] = np.array(raw_data[:,9])
  readable_data['ss8_probc'] = np.array(raw_data[:,10])
    
  np.savetxt(output_filename, readable_data, fmt="%d %c %c %c %4.3f %8.3f %8.3f %8.3f %8.3f %6.3f %6.3f %6.3f %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f", header='AA SS SS8 ASA Phi Psi Theta Tau HSEa_up HSEa_down CN P(3C) P(3E) P(3H) P(8G) P(8H) P(8I) P(8B) P(8E) P(8S) P(8T) P(8C)')

  
if __name__ == "__main__":
  import sys
  
  if len(sys.argv) == 2:
    pred_filename = sys.argv[1].split(' ')[0]
    seq_filename = sys.argv[1].split(' ')[1]
    readable_filename = sys.argv[1].split(' ')[2]
  else:
    print('invalid number of inputs')
    exit
  
  convert_raw_file(pred_filename, seq_filename, readable_filename)
  
  
