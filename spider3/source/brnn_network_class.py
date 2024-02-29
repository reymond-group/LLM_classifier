
####################################################  
import tensorflow as tf
import numpy as np


####################################################
# DEFINE SOME VARIABLES/PARAMETERS
####################################################
 
#def weight_variable(shape, mean=0.0, stddev=0.025):
#  initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)
#  return tf.Variable(initial)
#  
#def bias_variable(shape):
#  initial = tf.constant(0.01, shape=shape)
#  return tf.Variable(initial)

def fully_connected(input, weight_shape, bias_shape):
  # shape should be a list = [input_size, output_size]
  weights = tf.get_variable("weights", weight_shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.025))
  bias = tf.get_variable("bias", bias_shape, initializer=tf.constant_initializer(0.01))
  
  return tf.matmul(input, weights) + bias
   
def num_batches(num_vec, batch_size):
  incomplete_batch = 1 if np.mod(num_vec, batch_size) else 0
  return num_vec/batch_size+incomplete_batch
  


####################################################
# DEFINE NETWORK CLASSES
####################################################

class output_layer:
  # this layer's input is the output of a fully_connected_layer_with_dropout
  # the input shape will be 2D tensor of shape [SUM(sequences in batch lengths), output_size]
  # the output should be [SUM(sequences in batch lengths), output_size]
  def __init__(self, layer_input, input_size, output_size):
#    self.W = weight_variable([input_size, output_size])
#    self.b = bias_variable([output_size])
#    self.output = tf.matmul(layer_input,self.W) + self.b   
    self.output = fully_connected(layer_input, [input_size, output_size], [output_size])
    
   
  
class fully_connected_layer_with_dropout:
  # this layer's input is the output of a rnn_output_to_full_connected_reshape layer.
  # the input shape will be a 2D tensor of [batch_size*max_seq_len, output_size] <-- this doesn't seem right.
  # the output should be [batch_size*max_seq_len, output_size]
  def __init__(self, layer_input, input_size, layer_size, keep_prob, batch_norm=False, training=False):
    self.layer_input = layer_input
    self.keep_prob = keep_prob
    self.layer_size = layer_size
    self.input_size = input_size
    
    self.output = fully_connected(self.layer_input, [self.input_size, self.layer_size], [self.layer_size])
    
    if batch_norm==True:
      self.output = tf.layers.batch_normalization(self.output, axis=-1, scale=False, training=training)
    
#    # self.layer at this point is: 2D tensor of [batch_size*max_seq_len, 2*rnn_len]
#    self.output = tf.nn.relu(tf.matmul(self.layer_input, self.W) + self.b)
    self.output = tf.nn.relu(self.output)
    # self.output at this point is: 2D tensor of [batch_size*max_seq_len, output_size] 
    
    self.output = tf.nn.dropout(self.output, self.keep_prob)
    # self.output at this point is: 2D tensor of [batch_size*max_seq_len, output_size] 
    
class rnn_output_to_fully_connected_reshape:
  # this layer's input is the output of a (bi)rnn layer.
  # the input shape will be a 3D tensor of [batch_size, seq_len (n_steps), 2*rnn_layer_size (2*input_size)]
  # the output should be [batch_size*max_seq_len, output_size]
  # the output should be [SUM(seq_lengths), output_size]
  
  def __init__(self, layer_input, bool_length_mask, prev_layer_size):
    self.layer_size = prev_layer_size
    self.output = tf.boolean_mask(layer_input, bool_length_mask)

    
class brnn_layer:
  # input to this layer will be a list of n_steps length, with each element being a 2D
  # tensor of shape [batch_size, n_input]
  # output from this layer will a list of n_steps length, with each element being a 2D
  # tensor of shape [batch_size, 2*layer_size]
  def __init__ (self, ph_layer_input, ph_seq_lengths, n_input, layer_size, scope="RNN", cell_type='BasicLSTMCell', LSTM_internal_dropout_keep_prob=1.0):
    self.LSTM_internal_dropout_keep_prob = LSTM_internal_dropout_keep_prob
    self.ph_layer_input = ph_layer_input
    self.ph_seq_lengths = ph_seq_lengths
    self.n_input = n_input
    self.layer_size = layer_size
    
    # define basic lstm cells
    if cell_type == 'BasicLSTMCell':
      self.rnn_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.layer_size)
      self.rnn_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.layer_size)
    elif cell_type == 'LayerNormBasicLSTMCell':
      self.rnn_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.layer_size, dropout_keep_prob = self.LSTM_internal_dropout_keep_prob)
      self.rnn_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.layer_size, dropout_keep_prob = self.LSTM_internal_dropout_keep_prob)
    elif cell_type == 'GRUCell':
      self.lstm_fw_cell = tf.nn.rnn_cell.GRUCell(self.layer_size)
      self.lstm_bw_cell = tf.nn.rnn_cell.GRUCell(self.layer_size)
    

    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.rnn_fw_cell, self.rnn_bw_cell, self.ph_layer_input,
                                                                        sequence_length=self.ph_seq_lengths, dtype="float", 
                                                                        scope=scope)
    self.output = tf.concat(outputs, 2)
                                            
                                            
                                            
def bioinf_output_loss(output_type, true, pred, mask):
  # this functions takes a single output_type and returns the loss for that 
  # output type, given true and predicted values.
  # true and pred are tensors of shape [num_'frames', num_classes]
  # for output types such as SS, where we are doing classifiction, num_classes = 1 and we use
  # cross entropy loss.
  #
  # the true and pred being passed in should already be masked if they need to be.
  #
  # NOTE: tf version 1.0 changed order of cross entropy loss logit and label inputs.
  # OLD - tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)
  # NEW - tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)
  
  output_type = output_type.upper()
  
  if output_type == 'SS' or output_type == 'SS8':
    # apply the mask here. the loss function will work with labels of -1, however will not give correct gradients (???)
    masked_true = tf.to_int32(tf.boolean_mask(true, mask))
    masked_pred = tf.boolean_mask(pred, tf.reshape(mask, [-1]))    
#    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(masked_true, masked_pred))
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_true, logits=masked_pred))
  elif output_type == 'OMEGA': 
  # apply the mask here. the loss function will work with labels of -1, however will not give correct gradients (???)
    masked_true = tf.boolean_mask(true, mask)
    masked_pred = tf.boolean_mask(pred, mask)    
#    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(masked_true, masked_pred))
    loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(labels=masked_true, logits=masked_pred, pos_weight=300)) # 100/0.3 = 332.33..
  elif output_type == 'OMEGAPROLINE': 
  # apply the mask here. the loss function will work with labels of -1, however will not give correct gradients (???)
    masked_true = tf.boolean_mask(true, mask)
    masked_pred = tf.boolean_mask(pred, mask)    
#    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(masked_true, masked_pred))
    loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(labels=masked_true, logits=masked_pred, pos_weight=20)) # 100/0.3 = 332.33..
  elif output_type == 'ASA' or output_type == 'HSEA' or output_type == 'HSEB' or output_type == 'CN' or output_type == 'CN13' or output_type == 'THETA' or output_type == 'TAU' or output_type == 'PHI' or output_type == 'PSI' or output_type == 'TT' or output_type == 'PP':
    masked_true = tf.boolean_mask(true, mask)
    masked_pred = tf.sigmoid(tf.boolean_mask(pred, mask))
    loss = tf.nn.l2_loss(masked_true - masked_pred)
  elif output_type == 'TTPP':
    masked_true = tf.boolean_mask(true, mask)
    masked_pred = tf.sigmoid(tf.boolean_mask(pred, mask))
#    masked_pred = tf.tanh(tf.boolean_mask(pred, mask))
    loss = tf.nn.l2_loss(masked_true - masked_pred)
  else:
    print "ERROR INVALID OUTPUT TYPE", output_type
    
  return loss
    
    
def bioinf_output_nonlinearity(output_type, pred):
  # this function applies the nonlinear activation functions for the different output types.
  
  output_type = output_type.upper()
  if output_type == 'SS' or output_type == 'SS8':
    non_linear_output = tf.nn.softmax(pred)
  elif output_type == 'OMEGA' or output_type == 'OMEGAPROLINE':
    non_linear_output = tf.nn.sigmoid(pred)
  elif output_type == 'ASA' or output_type == 'HSEA' or output_type == 'HSEB' or output_type == 'CN' or output_type == 'CN13' or output_type == 'THETA' or output_type == 'TAU' or output_type == 'PHI' or output_type == 'PSI' or output_type == 'TT' or output_type == 'PP':
    non_linear_output = tf.sigmoid(pred)
  elif output_type == 'TTPP':
#    non_linear_output = tf.tanh(pred)
    non_linear_output = tf.sigmoid(pred)
  else:
    print "ERROR INVALID OUTPUT TYPE", output_type
    
    
  return non_linear_output

class brnn_network:
  def __init__(self, layer_sizes, output_types, output_index_true, output_index_pred,
                ph_network_input, n_input, 
#                n_steps,
                ph_seq_lengths, ph_seq_length_mask, ph_bool_length_mask, ph_network_output, ph_network_output_mask, ph_network_output_mask_encoded, n_classes, ph_keep_prob, LSTM_internal_dropout_keep_prob=1.0, cell_type='BasicLSTMCell'):   
    # network variables
    self.layer_sizes = layer_sizes
    self.output_types = output_types  # output_type is a list of stings for each output type.
    self.output_index_true = output_index_true # this is a list of lists of start and stop index for the true labels.
    self.output_index_pred = output_index_pred # this is a list of lists of start and stop index for the predicted labels.
    # note that the true labels will only contain an int for class labels (ie. the int will represent the class), while the
    # predicted labels will be using a one hot representation for class labels. This will cause the indexs to be different in the two cases.
    # the true and predicted labels will be the same shape for regression tasks.
    
    self.ph_seq_lengths = ph_seq_lengths
    self.n_input = n_input
    self.n_classes = n_classes
    self.ph_network_input = ph_network_input  # network input is the input data
    self.ph_network_output = ph_network_output # network output is the true labels
    self.ph_network_output_mask = ph_network_output_mask  # network output mask is a set of masks to remove undefined data points (ie X for SS class, or 180 for ASA values etc.)
    self.ph_network_output_mask_encoded = ph_network_output_mask_encoded  # network output mask of the same shape as the network's predictions
    self.ph_network_output_bool_mask = ph_network_output_mask>0  # network output mask is a set of masks to remove undefined data points (ie X for SS class, or 180 for ASA values etc.)
    self.ph_seq_len_mask = ph_seq_length_mask
    self.ph_bool_len_mask = ph_bool_length_mask
    self.ph_keep_prob = ph_keep_prob
    self.LSTM_internal_dropout_keep_prob = LSTM_internal_dropout_keep_prob # for use with LayerNormBasicLSTMCell
    self.cell_type = cell_type
    
    
    # network layers
    self.layer = []

    # reshape for LSTM inputs    
    self.ph_lstm_input = self.ph_network_input
    
    # lstm layers
    self.layer.append(brnn_layer(self.ph_lstm_input, 
                                 self.ph_seq_lengths, 
                                 self.n_input, 
#                                 self.n_steps,
                                 self.layer_sizes[0][0],
                                 scope="RNN1",
                                 cell_type = self.cell_type,
                                 LSTM_internal_dropout_keep_prob = self.LSTM_internal_dropout_keep_prob,
                                 ))
                                 
#    self.lstm_1_output = tf.pack(self.layer[-1].output)
    self.lstm_1_output = tf.nn.dropout(self.layer[-1].output, self.ph_keep_prob)
    
    self.layer.append(brnn_layer(self.lstm_1_output,
                                 self.ph_seq_lengths, 
                                 2*self.layer[-1].layer_size,
#                                 self.n_steps,
                                 self.layer_sizes[0][1],
                                 scope="RNN2",
                                 cell_type = self.cell_type,
                                 LSTM_internal_dropout_keep_prob = self.LSTM_internal_dropout_keep_prob,
                                 ))
                                 
    # reshape LSTM outputs    
    self.lstm_output = tf.nn.dropout(self.layer[-1].output, self.ph_keep_prob)
    
    # reshape layer
    self.layer.append(rnn_output_to_fully_connected_reshape(self.lstm_output, self.ph_bool_len_mask,
                                                            2*self.layer[-1].layer_size))
    
    # fully connected layers
    for fc_layer_num, n_hidden in enumerate(self.layer_sizes[1]):
      with tf.variable_scope("fully_connected"+str(fc_layer_num)):
        self.layer.append(fully_connected_layer_with_dropout(self.layer[-1].output,
                                                             self.layer[-1].layer_size,
                                                             n_hidden,
                                                             self.ph_keep_prob))
                            
    # output layer
    with tf.variable_scope("output_layer"):
#      self.layer.append(fully_connected(self.layer[-1].output, [self.layer[-1].layer_size, self.n_classes], [self.n_classes]))
      self.layer.append(output_layer(self.layer[-1].output, self.layer[-1].layer_size, self.n_classes))

    self.pred = self.layer[-1].output
    self.linear_output = self.pred
#    self.sigmoid_output = tf.sigmoid(self.pred)
    
    self.masked_pred = tf.multiply(self.ph_network_output_mask_encoded, self.pred)
    self.masked_network_output = tf.multiply(self.ph_network_output_mask, self.ph_network_output)
    
    temp_loss = []
    temp_non_linear_output = []
    for ind, output_type in enumerate(output_types):
          
      temp_loss.append(bioinf_output_loss(output_type, self.ph_network_output[:, self.output_index_true[ind][0]:self.output_index_true[ind][1]],
                                                       self.pred[:,self.output_index_pred[ind][0]:self.output_index_pred[ind][1]],
                                                       self.ph_network_output_bool_mask[:, self.output_index_true[ind][0]:self.output_index_true[ind][1]] ) )
                         
      temp_non_linear_output.append(bioinf_output_nonlinearity(output_type, self.linear_output[:, self.output_index_pred[ind][0]:self.output_index_pred[ind][1]]))
    
    self.non_linear_output = tf.concat(temp_non_linear_output, 1) 
    self.loss = tf.add_n(temp_loss)

    
  def get_predictions(self, input_feat, seq_len, batch_size=500, keep_prob=1.0):
    # this function will do a forward pass of the network and will return a set of 
    # predictions for each of the inputs.
    # the input to this function should be all of the input data you want predictions for
    # as well as the sequence masks that go along with the input data.
    
    
    for i in xrange(0, num_batches(len(input_feat), batch_size)):
      batch_ind = range(i*batch_size, np.minimum((i+1)*batch_size, len(input_feat)))
      batch_seq_lengths = [ seq_len[ind] for ind in batch_ind ]
      batch_max_length = max(batch_seq_lengths)
      batch_feat = np.array( [ np.concatenate((np.array(tmp), np.zeros((batch_max_length - tmp.shape[0], len(input_feat[0][0]))))) for tmp in [ input_feat[ind] for ind in batch_ind ] ] )
      batch_seq_len_mask = np.array( [ np.concatenate((np.ones(tmp), np.zeros(batch_max_length - tmp))) for tmp in batch_seq_lengths ] )

      feed_dict={self.ph_network_input: batch_feat,
                 self.ph_keep_prob: keep_prob,
                 self.LSTM_internal_dropout_keep_prob: keep_prob,
                 self.ph_seq_lengths: batch_seq_lengths,
                 self.ph_bool_len_mask: batch_seq_len_mask.astype(bool)}
      temp = self.non_linear_output.eval(feed_dict)
      
      # here would be a good place to convert from shape [SUM(batch lengths), # classes] to
      # a more useful shape [sequence length, # classes]
      
      if i == 0:
        np_output = temp
      else:
        np_output = np.concatenate((np_output, temp))    
    
    return np_output
    
    
    
  def get_loss(self, input_feat, true_labels, labels_mask, labels_mask_encoded, seq_len, batch_size=500, keep_prob=1.0):
    np_output = np.zeros(num_batches(len(input_feat), batch_size))
    
    for i in xrange(0, num_batches(len(input_feat), batch_size)):
      batch_ind = range(i*batch_size, np.minimum((i+1)*batch_size, len(input_feat)))
      batch_seq_lengths = [ seq_len[ind] for ind in batch_ind ]
      batch_max_length = max(batch_seq_lengths)
      batch_feat = np.array( [ np.concatenate((np.array(tmp), np.zeros((batch_max_length - tmp.shape[0], len(input_feat[0][0]))))) for tmp in [ input_feat[ind] for ind in batch_ind ] ] )
      batch_seq_len_mask = np.array( [ np.concatenate((np.ones(tmp), np.zeros(batch_max_length - tmp))) for tmp in batch_seq_lengths ] )
      batch_lab = np.concatenate( [ true_labels[ind] for ind in batch_ind])
      batch_lab_mask = np.concatenate( [ labels_mask[ind] for ind in batch_ind] )
      batch_lab_mask_encoded = np.concatenate( [ labels_mask_encoded[ind] for ind in batch_ind] )
      
      feed_dict={self.ph_network_input: batch_feat,
                 self.ph_seq_lengths: batch_seq_lengths,
                 self.ph_seq_len_mask: batch_seq_len_mask,
                 self.ph_bool_len_mask: batch_seq_len_mask.astype(bool),
                 self.ph_network_output: batch_lab,
                 self.ph_network_output_mask: batch_lab_mask,
                 self.ph_network_output_mask_encoded: batch_lab_mask_encoded,
                 self.ph_keep_prob: keep_prob}
      np_output[i] = self.loss.eval(feed_dict)
    return np.sum(np_output)
       
    
        
  def get_l2loss_from_predictions(self, network_output, true_labels, labels_mask):
    # this function takes the output from get_predictions and finds the loss of the network.
    # the network_output and true_labels are numpy arrays.
    #
    # this function is accurate enough. get_loss:2190.46899414 this function:2190.46884368
    
    error = true_labels - network_output
    masked_error = np.multiply(error, labels_mask)
    
    l2_loss = np.sum(np.square(masked_error)) / 2.0
    
    return l2_loss
    
  
     
