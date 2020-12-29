## Misc functions:
def get_pandas_path():
  import pandas
  pandas_path = pandas.__path__[0]
  return pandas_path

def run_missForest(data,Missing,maxits=20,n_estimators=20):
  import numpy as np
  from sklearn.ensemble import ExtraTreesRegressor
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  import time
  def mse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.power(xhat-xtrue,2)[~mask])
  
  xfull = (data - np.mean(data,0))/np.std(data,0)
  np.random.seed(1234)
  xmiss = np.copy(xfull)
  xmiss[Missing==0]=np.nan
  mask = np.isfinite(xmiss) # binary mask that indicates which values are missing
  
  t0_mf=time.time()
  missforest = IterativeImputer(max_iter=maxits, estimator=ExtraTreesRegressor(n_estimators=n_estimators))
  missforest.fit(xmiss)
  xhat_mf = missforest.transform(xmiss)
  time_mf=time.time()-t0_mf
  return{'time_mf': time_mf, 'MSE_mf': mse(xhat_mf,xfull,mask), 'xhat_mf': xhat_mf}

def run_meanImputation(data,Missing,maxits=None,n_estimators=None):
  import numpy as np
  from sklearn.impute import SimpleImputer
  import time
  def mse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.power(xhat-xtrue,2)[~mask])

  xfull = (data - np.mean(data,0))/np.std(data,0)
  np.random.seed(1234)
  xmiss = np.copy(xfull)
  xmiss[Missing==0]=np.nan
  mask = np.isfinite(xmiss) # binary mask that indicates which values are missing
  
  t0_mean=time.time()
  mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
  mean_imp.fit(xmiss)
  xhat_mean = mean_imp.transform(xmiss)
  time_mean=time.time()-t0_mean
  return{'time_mean': time_mean, 'MSE_mean': mse(xhat_mean,xfull,mask), 'xhat_mean': xhat_mean}

#################################################################################################

## HIVAE (Nazabal et al 2018) (main_scripts.py: uses function from read_functions.py):
# data_types: 
# bs: batch size, n_epochs: #epochs, train: training model flag, display: display trace every # epochs,
# n_save: save variables every n_save iters, restore: restore session
# restore: restore saved result, dim_latent_s-y: Dim of categorical/Z latent/Y latent space, dim_latent_y_partition: partition of Y latent space
# save_file: input string of 
# model_name: 'model_HIVAE_factorized' or 'model_HIVAE_inputDropout' (default)
def run_HIVAE(data,Missing,data_types,lr=1e-3,bs=200,n_epochs=500,train=1, display=100, n_save=1000, restore=0, dim_latent_s=10, dim_latent_z=2, dim_latent_y=10, dim_latent_y_partition=[], model_name="model_HIVAE_inputDropout",save_file="test"):
  import os
  import sys
  import argparse
  import tensorflow as tf
  import time
  import numpy as np
  import csv
  
  os.chdir("otherMethods/HIVAE")
  import read_functions as read_functions
  import graph_new as graph_new
  os.chdir("../..")
  # import parser_arguments

  def print_loss(epoch, start_time, avg_loss, avg_test_loglik, avg_KL_s, avg_KL_z):
      print("Epoch: [%2d]  time: %4.4f, train_loglik: %.8f, KL_z: %.8f, KL_s: %.8f, ELBO: %.8f, Test_loglik: %.8f"
            % (epoch, time.time() - start_time, avg_loss, avg_KL_z, avg_KL_s, avg_loss-avg_KL_z-avg_KL_s, avg_test_loglik))

  #Create a directoy for the save file
  if not os.path.exists('./HIVAE_Saved_Networks/' + save_file):
      os.makedirs('./HIVAE_Saved_Networks/' + save_file)

  network_file_name='./HIVAE_Saved_Networks/' + save_file + '/' + save_file +'.ckpt'
  log_file_name='./HIVAE_Saved_Network/' + save_file + '/log_file_' + save_file +'.txt'

  train_data, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.read_data(data, Missing, data_types)
  #Check batch size
  if bs > n_samples:
      bs = n_samples
  #Get an integer number of batches
  n_batches = int(np.floor(np.shape(train_data)[0]/bs))
  #Compute the real miss_mask
  miss_mask = np.multiply(miss_mask, true_miss_mask)

  #Creating graph
  sess_HVAE = tf.Graph()

  with sess_HVAE.as_default():
      tf_nodes = graph_new.HVAE_graph(model_name, data_types, bs,
                                  learning_rate=lr, z_dim=dim_latent_z, y_dim=dim_latent_y, s_dim=dim_latent_s, y_dim_partition=dim_latent_y_partition)

  ################### Running the VAE Training #################################

  with tf.Session(graph=sess_HVAE) as session:

      saver = tf.train.Saver()
      if(restore == 1):
          saver.restore(session, network_file_name)
          print("Model restored.")
      else:
          print('Initizalizing Variables ...')
          tf.global_variables_initializer().run()

      print('Training the HVAE ...')
      if(train == 1):
        
          start_time = time.time()
          # Training cycle
          loglik_epoch = []
          testloglik_epoch = []
          error_train_mode_global = []
          error_test_mode_global = []
          KL_s_epoch = []
          KL_z_epoch = []
          
          time_train=np.array([])
          time_impute=np.array([])
          time_rest=np.array([])
          ELBOs=np.array([])
          
          for epoch in range(n_epochs):
              avg_loss = 0.
              avg_KL_s = 0.
              avg_KL_z = 0.
              samples_list = []
              p_params_list = []
              q_params_list = []
              log_p_x_total = []
              log_p_x_missing_total = []

              # Annealing of Gumbel-Softmax parameter
              tau = np.max([1.0 - 0.01*epoch,1e-3])
              # tau = 1e-3
              tau2 = np.min([0.001*epoch,1.0])

              #Randomize the data in the mini-batches
              random_perm = np.random.permutation(range(np.shape(train_data)[0]))
              train_data_aux = train_data[random_perm,:]
              miss_mask_aux = miss_mask[random_perm,:]
              true_miss_mask_aux = true_miss_mask[random_perm,:]
              
              t_train=np.array([])
              t_impute=np.array([])
              for i in range(n_batches):

                  #Create inputs for the feed_dict
                  data_list, miss_list = read_functions.next_batch(train_data_aux, types_dict, miss_mask_aux, bs, index_batch=i)
                  #Delete not known data (input zeros)
                  data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[bs,1]) for i in range(len(data_list))]
                  #Create feed dictionary
                  feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                  feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                  feedDict[tf_nodes['miss_list']] = miss_list
                  feedDict[tf_nodes['tau_GS']] = tau
                  feedDict[tf_nodes['tau_var']] = tau2

                  t0_train=time.time()
                  #Running VAE
                  _,loss,KL_z,KL_s,samples,log_p_x,log_p_x_missing,p_params,q_params  = session.run([tf_nodes['optim'], tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'], tf_nodes['samples'],
                                                           tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],tf_nodes['p_params'],tf_nodes['q_params']],
                                                           feed_dict=feedDict)
                  t_train=np.append(t_train,time.time()-t0_train)
                  
                  t0_impute=time.time()
                  samples_test,log_p_x_test,log_p_x_missing_test,test_params  = session.run([tf_nodes['samples_test'],tf_nodes['log_p_x_test'],tf_nodes['log_p_x_missing_test'],tf_nodes['test_params']],
                                                               feed_dict=feedDict)
                  t_impute=np.append(t_impute,time.time()-t0_impute)

                  #Evaluate results on the imputation with mode, not on the samlpes!
                  samples_list.append(samples_test)
                  p_params_list.append(test_params)
                  #p_params_list.append(p_params)
                  q_params_list.append(q_params)
                  log_p_x_total.append(log_p_x_test)
                  log_p_x_missing_total.append(log_p_x_missing_test)

                  # Compute average loss
                  avg_loss += np.mean(loss)
                  avg_KL_s += np.mean(KL_s)
                  avg_KL_z += np.mean(KL_z)
                  
              time_train=np.append(time_train,np.sum(t_train))
              time_impute=np.append(time_impute,np.sum(t_impute))

              t0_rest=time.time()
              
              #Concatenate samples in arrays
              s_total, z_total, y_total, est_data = read_functions.samples_concatenation(samples_list)

              #Transform discrete variables back to the original values
              train_data_transformed = read_functions.discrete_variables_transformation(train_data_aux[:n_batches*bs,:], types_dict)
              est_data_transformed = read_functions.discrete_variables_transformation(est_data, types_dict)
              est_data_imputed = read_functions.mean_imputation(train_data_transformed, miss_mask_aux[:n_batches*bs,:], types_dict)

              #est_data_transformed[np.isinf(est_data_transformed)] = 1e20

              #Create global dictionary of the distribution parameters
              p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, types_dict, dim_latent_z, dim_latent_s)
              q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list,  dim_latent_z, dim_latent_s)

              #Number of clusters created
              cluster_index = np.argmax(q_params_complete['s'],1)
              cluster = np.unique(cluster_index)
              print('Clusters: ' + str(len(cluster)))

              #Compute mean and mode of our loglik models
              loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'],types_dict)
              #loglik_mean[np.isinf(loglik_mean)] = 1e20

              #Try this for the errors
              error_train_mean, error_test_mean = read_functions.error_computation(train_data_transformed, loglik_mean, types_dict, miss_mask_aux[:n_batches*bs,:])
              error_train_mode, error_test_mode = read_functions.error_computation(train_data_transformed, loglik_mode, types_dict, miss_mask_aux[:n_batches*bs,:])
              error_train_samples, error_test_samples = read_functions.error_computation(train_data_transformed, est_data_transformed, types_dict, miss_mask_aux[:n_batches*bs,:])
              error_train_imputed, error_test_imputed = read_functions.error_computation(train_data_transformed, est_data_imputed, types_dict, miss_mask_aux[:n_batches*bs,:])

              #Compute test-loglik from log_p_x_missing
              log_p_x_total = np.transpose(np.concatenate(log_p_x_total,1))
              log_p_x_missing_total = np.transpose(np.concatenate(log_p_x_missing_total,1))
              avg_test_loglik = np.sum(log_p_x_missing_total)/(np.sum(1.0-miss_mask_aux)+1e-100)

              # Display logs per epoch step
              if epoch % display == 0:
                  print_loss(epoch, start_time, avg_loss/n_batches, avg_test_loglik, avg_KL_s/n_batches, avg_KL_z/n_batches)
                  print('Test error mode: ' + str(np.round(np.mean(error_test_mode),3)))
                  print("")

              #Compute train and test loglik per variables
              loglik_per_variable = np.sum(log_p_x_total,0)/(np.sum(miss_mask_aux,0)+1e-100) # add very small value --> avoid NaNs
              loglik_per_variable_missing = np.sum(log_p_x_missing_total,0)/(np.sum(1.0-miss_mask_aux,0)+1e-100)

              #Store evolution of all the terms in the ELBO
              loglik_epoch.append(loglik_per_variable)
              testloglik_epoch.append(loglik_per_variable_missing)
              KL_s_epoch.append(avg_KL_s/n_batches)
              KL_z_epoch.append(avg_KL_z/n_batches)
              error_train_mode_global.append(error_train_mode)
              error_test_mode_global.append(error_test_mode)
              LB = (avg_loss/n_batches)-(avg_KL_z/n_batches)-(avg_KL_s/n_batches)
              ELBOs=np.append(ELBOs,LB)


              if epoch % n_save == 0:
                  print('Saving Variables ...')
                  save_path = saver.save(session, network_file_name)
                  
              time_rest=np.append(time_rest,time.time()-t0_rest)


          print('Training Finished ...')
          end_time=time.time()
          time_HIVAE_train=end_time-start_time

          #Saving needed variables in csv
          if not os.path.exists('./HIVAE_Results_csv/' + save_file):
              os.makedirs('./HIVAE_Results_csv/' + save_file)

          with open('HIVAE_Results_csv/' + save_file + '/' + save_file + '_loglik.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(loglik_epoch)

          with open('HIVAE_Results_csv/' + save_file + '/' + save_file + '_testloglik.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(testloglik_epoch)

          with open('HIVAE_Results_csv/' + save_file + '/' + save_file + '_KL_s.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(np.reshape(KL_s_epoch,[-1,1]))

          with open('HIVAE_Results_csv/' + save_file + '/' + save_file + '_KL_z.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(np.reshape(KL_z_epoch,[-1,1]))

          with open('HIVAE_Results_csv/' + save_file + '/' + save_file + '_train_error.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(error_train_mode_global)

          with open('HIVAE_Results_csv/' + save_file + '/' + save_file + '_test_error.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(error_test_mode_global)

          # Save the variables to disk at the end
          save_path = saver.save(session, network_file_name)
          
          train_params = {'lr': lr, 'bs': bs, 'n_epochs': n_epochs, 'dim_latent_s': dim_latent_s, 'dim_latent_z': dim_latent_z, 'dim_latent_y': dim_latent_y, 'model_name': model_name}
          return {'train_params': train_params, 'ELBOs': ELBOs,'time_train':time_train,'time_impute':time_impute,'time_rest':time_rest,'time': time_HIVAE_train,'samples_list': samples_list,'p_params_list': p_params_list,'q_params_list': q_params_list,'s_total': s_total, 'z_total': z_total, 'y_total': y_total, 'est_data': est_data,'ll': loglik_epoch, 'test_ll': testloglik_epoch, 'KL_s': KL_s_epoch, 'KL_z': KL_z_epoch, 'err_train_mode_global': error_train_mode_global, 'err_train_mean': error_train_mean, 'err_train_imputed': error_train_imputed, 'err_test_mode_global': error_test_mode_global, 'err_test_mean': error_test_mean, 'err_test_imputed': error_test_imputed, 'train_x_transf': train_data_transformed, 'est_x_transf': est_data_transformed, 'est_x_imputed': est_data_imputed}
        

      #Test phase
      else:
          start_time = time.time()
          # Training cycle

          #f_toy2, ax_toy2 = plt.subplots(4,4,figsize=(8, 8))
          loglik_epoch = []
          testloglik_epoch = []
          error_train_mode_global = []
          error_test_mode_global = []
          error_imputed_global = []
          est_data_transformed_total = []

          #Only one epoch needed, since we are doing mode imputation
          for epoch in range(n_epochs):
              avg_loss = 0.
              avg_KL_s = 0.
              avg_KL_y = 0.
              avg_KL_z = 0.
              samples_list = []
              p_params_list = []
              q_params_list = []
              log_p_x_total = []
              log_p_x_missing_total = []

              label_ind = 2

              # Constant Gumbel-Softmax parameter (where we have finished the annealing)
              tau = 1e-3
              #tau = 1.0

              # Randomize the data in the mini-batches
              #random_perm = np.random.permutation(range(np.shape(train_data)[0]))
              # random_perm = range(np.shape(train_data)[0])
              random_perm = range(n_batches*bs)
              train_data_aux = train_data[random_perm,:]
              miss_mask_aux = miss_mask[random_perm,:]
              true_miss_mask_aux = true_miss_mask[random_perm,:]

              for i in range(n_batches):

                  #Create train minibatch
                  data_list, miss_list = read_functions.next_batch(train_data_aux, types_dict, miss_mask_aux, bs,
                                                                   index_batch=i)
                  #print(np.mean(data_list[0],0))

                  #Delete not known data
                  data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[bs,1]) for i in range(len(data_list))]


                  #Create feed dictionary
                  feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                  feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                  feedDict[tf_nodes['miss_list']] = miss_list
                  feedDict[tf_nodes['tau_GS']] = tau

                  #Get samples from the model
                  loss,KL_z,KL_s,samples,log_p_x,log_p_x_missing,p_params,q_params  = session.run([tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'], tf_nodes['samples'],
                                                               tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],tf_nodes['p_params'],tf_nodes['q_params']],
                                                               feed_dict=feedDict)

                  samples_test,log_p_x_test,log_p_x_missing_test,test_params  = session.run([tf_nodes['samples_test'],tf_nodes['log_p_x_test'],tf_nodes['log_p_x_missing_test'],tf_nodes['test_params']],
                                                               feed_dict=feedDict)


                  samples_list.append(samples_test)
                  p_params_list.append(test_params)
                  q_params_list.append(q_params)
                  log_p_x_total.append(log_p_x_test)
                  log_p_x_missing_total.append(log_p_x_missing_test)

                  # Compute average loss
                  avg_loss += np.mean(loss)
                  avg_KL_s += np.mean(KL_s)
                  avg_KL_z += np.mean(KL_z)

              #Separate the samples from the batch list
              s_aux, z_aux, y_total, est_data = read_functions.samples_concatenation(samples_list)

              #Transform discrete variables to original values
              train_data_transformed = read_functions.discrete_variables_transformation(train_data_aux[:n_batches*bs,:], types_dict)
              est_data_transformed = read_functions.discrete_variables_transformation(est_data, types_dict)
              est_data_imputed = read_functions.mean_imputation(train_data_transformed, miss_mask_aux[:n_batches*bs,:], types_dict)

              #Create global dictionary of the distribution parameters
              p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, types_dict, dim_latent_z, dim_latent_s)
              q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list,  dim_latent_z, dim_latent_s)

              #Number of clusters created
              cluster_index = np.argmax(q_params_complete['s'],1)
              cluster = np.unique(cluster_index)
              print('Clusters: ' + str(len(cluster)))

              #Compute mean and mode of our loglik models
              loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'],types_dict)

              #Try this for the errors
              error_train_mean, error_test_mean = read_functions.error_computation(train_data_transformed, loglik_mean, types_dict, miss_mask_aux[:n_batches*bs,:])
              error_train_mode, error_test_mode = read_functions.error_computation(train_data_transformed, loglik_mode, types_dict, miss_mask_aux[:n_batches*bs,:])
              error_train_samples, error_test_samples = read_functions.error_computation(train_data_transformed, est_data_transformed, types_dict, miss_mask_aux[:n_batches*bs,:])
              error_train_imputed, error_test_imputed = read_functions.error_computation(train_data_transformed, est_data_imputed, types_dict, miss_mask_aux[:n_batches*bs,:])

              # Compute test-loglik from log_p_x_missing
              log_p_x_missing_total = np.transpose(np.concatenate(log_p_x_missing_total,1))
              avg_test_loglik = np.sum(log_p_x_missing_total)/(np.sum(1.0-miss_mask_aux)+1e-100)

              # Display logs per epoch step
              if epoch % display == 0:
                  print(np.round(error_test_mode,3))
                  print('Test error mode: ' + str(np.round(np.mean(error_test_mode),3)))
                  print("")

              #Plot evolution of test loglik
              loglik_per_variable = np.sum(np.concatenate(log_p_x_total,1),1)/(np.sum(miss_mask,0)+1e-100)
              loglik_per_variable_missing = np.sum(log_p_x_missing_total,0)/(np.sum(1.0-miss_mask,0)+1e-100)

              loglik_epoch.append(loglik_per_variable)
              testloglik_epoch.append(loglik_per_variable_missing)

              print('Test loglik: ' + str(np.round(np.mean(loglik_per_variable_missing),3)))


              #Re-run test error mode
              error_train_mode_global.append(error_train_mode)
              error_test_mode_global.append(error_test_mode)
              error_imputed_global.append(error_test_imputed)

              #Store data samples
              est_data_transformed_total.append(est_data_transformed)

          #Compute the data reconstruction
          data_reconstruction = train_data_transformed * miss_mask_aux[:n_batches*bs,:] + \
                                      np.round(loglik_mode,3) * (1 - miss_mask_aux[:n_batches*bs,:])
    
          train_data_transformed = train_data_transformed[np.argsort(random_perm)]
          data_reconstruction = data_reconstruction[np.argsort(random_perm)]
          
          end_time=time.time()
          time_HIVAE_test=end_time-start_time


          if not os.path.exists('./HIVAE_Results/' + save_file):
              os.makedirs('./HIVAE_Results/' + save_file)

          with open('HIVAE_Results/' + save_file + '/' + save_file + '_data_reconstruction.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(data_reconstruction)
          with open('HIVAE_Results/' + save_file + '/' + save_file + '_data_true.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(train_data_transformed)


          #Saving needed variables in csv
          if not os.path.exists('./HIVAE_Results_test_csv/' + save_file):
              os.makedirs('./HIVAE_Results_test_csv/' + save_file)

          #Train loglik per variable
          with open('HIVAE_Results_test_csv/' + save_file + '/' + save_file + '_loglik.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(loglik_epoch)

          #Test loglik per variable
          with open('HIVAE_Results_test_csv/' + save_file + '/' + save_file + '_testloglik.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(testloglik_epoch)

          #Train NRMSE per variable
          with open('HIVAE_Results_test_csv/' + save_file + '/' + save_file + '_train_error.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(error_train_mode_global)

          #Test NRMSE per variable
          with open('HIVAE_Results_test_csv/' + save_file + '/' + save_file + '_test_error.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows(error_test_mode_global)

          #Number of clusters
          with open('HIVAE_Results_test_csv/' + save_file + '/' + save_file + '_clusters.csv', "w") as f:
              writer = csv.writer(f)
              writer.writerows([[len(cluster)]])
              
          return {'time': time_HIVAE_test, 'cluster': cluster,'samples_list': samples_list,'p_params_list': p_params_list,'q_params_list': q_params_list, 's_aux': s_aux, 'z_aux': z_aux, 'y_total': y_total, 'est_data': est_data,'data_reconstructed': data_reconstruction, 'train_data_transformed': train_data_transformed, 'll': loglik_epoch, 'test_ll': testloglik_epoch, 'err_train_mode_global': error_train_mode_global, 'err_train_mean': error_train_mean, 'err_train_imputed': error_train_imputed, 'err_test_mode_global': error_test_mode_global, 'err_test_mean': error_test_mean, 'err_test_imputed': error_test_imputed, 'err_imp_global': error_imputed_global, 'mean_loss': avg_loss, 'mean_KL_s': avg_KL_s, 'mean_KL_z': avg_KL_z}

#################################################################################################

## VAEAC (Ivanov et al 2019)
# one_hot_max_sizes (list: c(ints)): The space-separated list of one-hot max sizes for categorical features and 
# 0 or 1 for real-valued ones. A categorical feature is supposed to be a column of integers 
# from 0 to K-1, where K is one-hot max size for the feature. The length of the list
# must be equal to the number of columns in the data
# validation_ratio: proportion of samples to include in validation set
# validations_per_epoch: integer number of IWAE estimations on the validation set per one epoch on the training set
# validation_iwae_n_samples: number of samples per object to estimate IWAE on the validation set
# restore: True/False for restoring previously trained VAE
# output_file: file name where you want to save results (tsv)
def run_VAEAC(data,one_hot_max_sizes,norm_mean,norm_std,h,n_hidden_layers,dim_z,bs,lr,output_file,train=1,saved_model=None,saved_networks=None,n_epochs=500,n_imputations=5,validation_ratio=0.2,validations_per_epoch=1,validation_iwae_n_samples=25,restore=False):
  from argparse import ArgumentParser
  from copy import deepcopy
  from importlib import import_module
  from math import ceil
  from os.path import exists, join
  from sys import stderr

  import numpy as np
  import torch
  from torch.utils.data import DataLoader
  from tqdm import tqdm
  import os
  
  #from datasets import compute_normalization    # did it manually in R --> fed in
  os.chdir("otherMethods/vaeac")
  from imputation_networks import get_imputation_networks
  from train_utils import extend_batch, get_validation_iwae
  from VAEAC import VAEAC
  os.chdir("../..")
  import time
  
  import warnings
  warnings.simplefilter("ignore") # ignore the deprecation warnings

  # Read and normalize input data
  raw_data = torch.from_numpy(data).float()
  # norm_mean, norm_std = compute_normalization(raw_data, one_hot_max_sizes)
  
  norm_mean = torch.from_numpy(norm_mean).float()
  norm_std = torch.max(torch.from_numpy(norm_std).float(), torch.tensor(1e-9))
  
  # print(raw_data.shape)
  # print(norm_mean.shape)
  # print(norm_std.shape)
  data = (raw_data - norm_mean[None]) / norm_std[None]
  
  # Default parameters which are not supposed to be changed from user interface
  use_cuda = torch.cuda.is_available()
  verbose = True
  # Non-zero number of workers cause nasty warnings because of some bug in
  # multiprocess library. It might be fixed now, but anyway there is no need
  # to have a lot of workers for dataloader over in-memory tabular data.
  num_workers = 0
  
  if train==1:
    # design all necessary networks and learning parameters for the dataset
    networks = get_imputation_networks(one_hot_max_sizes,h,n_hidden_layers,dim_z,bs,lr)
  
    # build VAEAC on top of returned network, optimizer on top of VAEAC,
    # extract optimization parameters and mask generator
    model = VAEAC(
        networks['reconstruction_log_prob'],
        networks['proposal_network'],
        networks['prior_network'],
        networks['generative_network']
    )
    if use_cuda:
        model = model.cuda()
    optimizer = networks['optimizer'](model.parameters())
    batch_size = networks['batch_size']
    mask_generator = networks['mask_generator']
    vlb_scale_factor = networks.get('vlb_scale_factor', 1)
    
    # train-validation split
    val_size = ceil(len(data) * validation_ratio)
    val_indices = np.random.choice(len(data), val_size, False)
    val_indices_set = set(val_indices)
    train_indices = [i for i in range(len(data)) if i not in val_indices_set]
    train_data = data[train_indices]
    val_data = data[val_indices]
    
    # initialize dataloaders
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=False)
    
    # number of batches after which it is time to do validation
    validation_batches = ceil(len(dataloader) / validations_per_epoch)
    
    # a list of validation IWAE estimates
    validation_iwae = []
    # a list of running variational lower bounds on the train set
    train_vlb = []
    # the length of two lists above is the same because the new
    # values are inserted into them at the validation checkpoints only
    
    # best model state according to the validation IWAE
    best_state = None
    
    # main train loop
    for epoch in range(n_epochs):
    
        iterator = dataloader
        avg_vlb = 0
        if verbose:
            print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
            iterator = tqdm(iterator)
    
        # one epoch
        for i, batch in enumerate(iterator):
    
            # the time to do a checkpoint is at start and end of the training
            # and after processing validation_batches batches
            if any([
                        i == 0 and epoch == 0,
                        i % validation_batches == validation_batches - 1,
                        i + 1 == len(dataloader)
                    ]):
                val_iwae = get_validation_iwae(val_dataloader, mask_generator,
                                               batch_size, model,
                                               validation_iwae_n_samples,
                                               verbose)
                validation_iwae.append(val_iwae)
                train_vlb.append(avg_vlb)
    
                # if current model validation IWAE is the best validation IWAE
                # over the history of training, the current state
                # is saved to best_state variable
                if max(validation_iwae[::-1]) <= val_iwae:
                    best_state = deepcopy({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'validation_iwae': validation_iwae,
                        'train_vlb': train_vlb,
                    })
    
                if verbose:
                    print(file=stderr)
                    print(file=stderr)
    
            # if batch size is less than batch_size, extend it with objects
            # from the beginning of the dataset
            batch = extend_batch(batch, dataloader, batch_size)
    
            # generate mask and do an optimizer step over the mask and the batch
            mask = mask_generator(batch)
            optimizer.zero_grad()
            if use_cuda:
                batch = batch.cuda()
                mask = mask.cuda()
            vlb = model.batch_vlb(batch, mask).mean()
            (-vlb / vlb_scale_factor).backward()
            optimizer.step()
    
            # update running variational lower bound average
            avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
            if verbose:
                iterator.set_description('Train VLB: %g' % avg_vlb)
    
    # if use doesn't set use_last_checkpoint flag,
    # use the best model according to the validation IWAE
    if not restore:
        model.load_state_dict(best_state['model_state_dict'])
    
    train_params = {'h':h, 'n_hidden_layers':n_hidden_layers, 'dim_z':dim_z, 'bs':bs, 'lr':lr, 'n_epochs': n_epochs, 'n_imputations':n_imputations, 'validation_ratio':validation_ratio, 'validations_per_epoch':validations_per_epoch,'validation_iwae_n_samples':validation_iwae_n_samples}
    return {'saved_model': model, 'saved_networks': networks,'train_params': train_params, 'LB':avg_vlb,'best_state': best_state}

  elif train==0:
    model = saved_model
    networks = saved_networks
    
    # one epoch through to get avg_vlb
    #optimizer = networks['optimizer'](model.parameters())
    batch_size = networks['batch_size']
    mask_generator = networks['mask_generator']

    # train-validation split
    #val_size = ceil(len(data) * 0)
    #val_indices = np.random.choice(len(data), val_size, False)
    #val_indices_set = set(val_indices)
    #train_indices = [i for i in range(len(data)) if i not in val_indices_set]
    #train_data = data[train_indices]
    train_data = data  ## "Train" data: all data --> just for vlb quantity

    # initialize dataloader
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=False)
    
    # best model state according to the validation IWAE
    best_state = None
    n_epochs=1
    for epoch in range(n_epochs):
    
        iterator = dataloader
        avg_vlb = 0
        if verbose:
            print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
            iterator = tqdm(iterator)
    
        # one epoch
        for i, batch in enumerate(iterator):
    
            # if batch size is less than batch_size, extend it with objects
            # from the beginning of the dataset
            batch = extend_batch(batch, dataloader, batch_size)
    
            # generate mask and do an optimizer step over the mask and the batch
            mask = mask_generator(batch)
            if use_cuda:
                batch = batch.cuda()
                mask = mask.cuda()
            vlb = model.batch_vlb(batch, mask).mean()

            # update running variational lower bound average
            avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
            if verbose:
                iterator.set_description('Train VLB: %g' % avg_vlb)
    
  
    # build dataloader for the whole input data
    dataloader = DataLoader(data, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            drop_last=False)
  
    # prepare the store for the imputations
    results = []
    for i in range(n_imputations):
        results.append([])
    
    iterator = dataloader
    if verbose:
        iterator = tqdm(iterator)
    
    # impute missing values for all input data
    for batch in iterator:
    
        # if batch size is less than batch_size, extend it with objects
        # from the beginning of the dataset
        batch_extended = torch.tensor(batch)
        batch_extended = extend_batch(batch_extended, dataloader, batch_size)
    
        if use_cuda:
            batch = batch.cuda()
            batch_extended = batch_extended.cuda()
    
        # compute the imputation mask
        mask_extended = torch.isnan(batch_extended).float()
    
        # compute imputation distributions parameters
        with torch.no_grad():
            samples_params = model.generate_samples_params(batch_extended,
                                                           mask_extended,
                                                           n_imputations)
            samples_params = samples_params[:batch.shape[0]]
    
        # make a copy of batch with zeroed missing values
        mask = torch.isnan(batch)
        batch_zeroed_nans = torch.tensor(batch)
        batch_zeroed_nans[mask] = 0
    
        # impute samples from the generative distributions into the data
        # and save it to the results
        for i in range(n_imputations):
            sample_params = samples_params[:, i]
            sample = networks['sampler'](sample_params)
            sample[(~mask).byte()] = 0
            sample += batch_zeroed_nans
            results[i].append(torch.tensor(sample, device='cpu'))
    
    # concatenate all batches into one [n x K x D] tensor,
    # where n in the number of objects, K is the number of imputations
    # and D is the dimensionality of one object
    for i in range(len(results)):
        results[i] = torch.cat(results[i]).unsqueeze(1)
    result = torch.cat(results, 1)
    
    # reshape result, undo normalization and save it
    result = result.view(result.shape[0] * result.shape[1], result.shape[2])
    result = result * norm_std[None] + norm_mean[None]
    
    if output_file:
      np.savetxt(output_file, result.numpy(), delimiter='\t')
    if use_cuda:
      sample=sample.detach().cpu().clone().numpy()
      samples_params=samples_params.detach().cpu().clone().numpy()
      result=result.detach().cpu().clone().numpy()
      data=data.detach().cpu().clone().numpy()
    else:
      sample=sample.numpy()
      samples_params=samples_params.numpy()
      result=result.numpy()
      data=data.numpy()
    
    train_params = {'h':h, 'n_hidden_layers':n_hidden_layers, 'dim_z':dim_z, 'bs':bs, 'lr':lr, 'n_epochs': n_epochs, 'n_imputations':n_imputations, 'validation_ratio':validation_ratio, 'validations_per_epoch':validations_per_epoch,'validation_iwae_n_samples':validation_iwae_n_samples}
    return {'train_params': train_params, 'LB':avg_vlb, 'sample': sample,'samples_params': samples_params,'best_model': model,'result': result, 'data': data}




## MIWAE (Mattei et al 2019):
def run_MIWAE(data,Missing,norm_means,norm_sds,n_hidden_layers=2,dec_distrib="Normal",train=1,saved_model=None,h=10,sigma="relu",bs = 64,n_epochs = 2002,lr=0.001,niw=20,dim_z=5,L=20,trace=False):
  # L: number of MC samples used in imputation
  import torch
  #import torchvision
  import torch.nn as nn
  import numpy as np
  import scipy.stats
  import scipy.io
  import scipy.sparse
  from scipy.io import loadmat
  import pandas as pd
  from matplotlib.backends.backend_pdf import PdfPages
  import matplotlib.pyplot as plt
  import torch.distributions as td

  from torch import nn, optim
  from torch.nn import functional as F
  # from torchvision import datasets, transforms
  # from torchvision.utils import save_image

  import time

  def mse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return {'miss':np.mean(np.power(xhat-xtrue,2)[~mask]),'obs':np.mean(np.power(xhat-xtrue,2)[mask])}
  
  time0 = time.time()
    
  # xfull = (data - np.mean(data,0))/np.std(data,0)
  xfull = (data - norm_means)/norm_sds
  n = xfull.shape[0] # number of observations
  p = xfull.shape[1] # number of features
  
  np.random.seed(1234)
  
  xmiss = np.copy(xfull)
  xmiss[Missing==0]=np.nan
  mask = np.isfinite(xmiss) # binary mask that indicates which values are missing
  mask0 = np.copy(mask)
  
  xhat_0 = np.copy(xmiss)
  xhat_0[np.isnan(xmiss)] = 0
  
  d = dim_z # dimension of the latent space
  K = niw # number of IS during training
  
  bs = min(bs,n)
  impute_bs = min(bs, n)

  p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)     # THIS IS NORMAL vs. student T used in CPU version!!
  if (dec_distrib=="Normal"): num_dec_params=2
  elif (dec_distrib=="StudentT"): num_dec_params=3
  
  if (sigma=="relu"): act_fun=torch.nn.ReLU()
  elif (sigma=="elu"): act_fun=torch.nn.ELU()
  
  def network_maker(act_fun, n_hidden_layers, in_h, h, out_h, dropout=False):
    if n_hidden_layers==0:
      layers = [ nn.Linear(in_h, out_h), ]
    elif n_hidden_layers>0:
      layers = [ nn.Linear(in_h , h), act_fun, ]
      for i in range(n_hidden_layers-1):
        layers.append( nn.Linear(h, h), )
        layers.append( act_fun, )
      layers.append(nn.Linear(h, out_h))
    elif n_hidden_layers<0:
      raise Exception("n_hidden_layers must be >= 0")
    if dropout:
      layers.insert(0, nn.Dropout())
    model = nn.Sequential(*layers)
    return model
  
  encoder = network_maker(act_fun, n_hidden_layers, p, h, 2*d, False)
  decoder = network_maker(act_fun, n_hidden_layers, d, h, num_dec_params*p, False)
  
  # decoder = nn.Sequential(
  #   torch.nn.Linear(d, h),
  #   torch.nn.ReLU(),
  #   torch.nn.Linear(h, num_dec_params*p),
  # )
  # encoder = nn.Sequential(
  #   torch.nn.Linear(p, h),
  #   torch.nn.ReLU(),
  #   torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
  # )
  
  encoder.cuda() # we'll use the GPU
  decoder.cuda()
  
  def miwae_loss(iota_x,mask):
    batch_size = iota_x.shape[0]
    out_encoder = encoder(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
    
    zgivenx = q_zgivenxobs.rsample([K])
    zgivenx_flat = zgivenx.reshape([K*batch_size,d])
    
    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
    if dec_distrib=="StudentT":
      all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3
    
    data_flat = torch.Tensor.repeat(iota_x,[K,1]).reshape([-1,1])
    tiledmask = torch.Tensor.repeat(mask,[K,1])
    
    if dec_distrib=="Normal":
      all_log_pxgivenz_flat = torch.distributions.Normal(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1])).log_prob(data_flat)
      params_x={'mean':all_means_obs_model,'sd':all_scales_obs_model}
    elif dec_distrib=="StudentT":
      all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
      params_x={'mean':all_means_obs_model,'sd':all_scales_obs_model,'df':all_degfreedom_obs_model}
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p])     # p(x|z) : Product of 1-D student's T. q(z|x) : MV-Gaussian
    
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)
    
    # neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))
    neg_bound = -torch.sum(torch.logsumexp(logpxobsgivenz + logpz - logq,0))  # average this after summing minibatches
    params_z={'mean':out_encoder[..., :d], 'sd':torch.nn.Softplus()(out_encoder[..., d:(2*d)])}
    return{'neg_bound':neg_bound, 'params_x': {'mean': params_x['mean'].detach(), 'sd': params_x['sd'].detach()}, 'params_z': {'mean': params_z['mean'].detach(), 'sd': params_z['sd'].detach()}}
  
  optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=lr)
  
  def miwae_impute(iota_x,mask,L):
    batch_size = iota_x.shape[0]
    out_encoder = encoder(iota_x)

    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
    
    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L*batch_size,d])
    
    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
    if dec_distrib=="StudentT":
      all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3
    
    data_flat = torch.Tensor.repeat(iota_x,[L,1]).reshape([-1,1]).cuda()
    tiledmask = torch.Tensor.repeat(mask,[L,1]).cuda()
    
    if dec_distrib=="Normal":
      all_log_pxgivenz_flat = td.Normal(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1])).log_prob(data_flat)
      xgivenz = td.Independent(td.Normal(loc=all_means_obs_model, scale=all_scales_obs_model),1)
    elif dec_distrib=="StudentT":
      all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
      xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,p])
    
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)
    
    imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
    xms = xgivenz.sample().reshape([L,batch_size,p])
    xm=torch.einsum('ki,kij->ij', imp_weights, xms) 
    return {'xm': xm.detach(), 'imp_weights': imp_weights.detach(),'zgivenx_flat': zgivenx_flat.detach()}
  def weights_init(layer):
    if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
  
  miwae_loss_train=np.array([])
  mse_train_miss=np.array([])
  mse_train_obs=np.array([])
  bs = bs # batch size
  n_epochs = n_epochs
  xhat = np.copy(xhat_0) # This will be out imputed data matrix
  
  trace_ids = np.concatenate([np.where(Missing[:,0]==0)[0][0:2],np.where(Missing[:,0]==1)[0][0:2]])
  
  if trace:
    print(xhat_0[trace_ids])

  encoder.apply(weights_init)
  decoder.apply(weights_init)
  
  time_train=[]
  time_impute=[]
  MIWAE_LB_epoch=[]
  if train==1:
    # Training+Imputing
    for ep in range(1,n_epochs):
      perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      batches_data = np.array_split(xhat_0[perm,], n/bs)
      batches_mask = np.array_split(mask0[perm,], n/bs)
      t0_train=time.time()
      splits = np.array_split(perm, n/bs)
      batches_loss = []
      loss_fits = []
      for it in range(len(batches_data)):
        optimizer.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()
        b_data = torch.from_numpy(batches_data[it]).float().cuda()
        b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
        loss_fit = miwae_loss(iota_x = b_data,mask = b_mask)
        loss = loss_fit['neg_bound']
        
        batches_loss = np.append(batches_loss, loss.cpu().data.numpy())
        loss.backward()
        
        loss_fit.pop("neg_bound")  # remove loss to not save computational graph associated with it
        loss_fits = np.append(loss_fits, {'loss_fit': loss_fit, 'obs_ids': splits[it]})
        optimizer.step()
      time_train=np.append(time_train,time.time()-t0_train)
      # loss_fit=miwae_loss(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda())
      # MIWAE_LB=(-np.log(K)-loss_fit['neg_bound'].cpu().data.numpy())
      total_loss = -np.sum(batches_loss)
      MIWAE_LB = total_loss / (K*n) - np.log(K)
      MIWAE_LB_epoch = np.append(MIWAE_LB_epoch,MIWAE_LB)
      if ep % 100 == 1:
        print('Epoch %g' %ep)
        print('MIWAE likelihood bound  %g' %MIWAE_LB) # Gradient step
        # if trace:
        #   print(loss_fit['params_x']['mean'][trace_ids])
        #   print(loss_fit['params_x']['sd'][trace_ids])
        ### Now we do the imputation
        t0_impute=time.time()
        
        # xhat_fit=miwae_impute(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda(),L=L)
        # xhat[~mask] = xhat_fit['xm'].cpu().data.numpy()[~mask]
        # #xhat = xhat_fit['xm'].cpu().data.numpy()  # observed values are not imputed
        batches_data = np.array_split(xhat_0, n/impute_bs)
        batches_mask = np.array_split(mask0, n/impute_bs)
        splits = np.array_split(range(n),n/impute_bs)
        xhat_fits=[]
        for it in range(len(batches_data)):
          b_data = torch.from_numpy(batches_data[it]).float().cuda()
          b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
          xhat_fit=miwae_impute(iota_x = b_data, mask = b_mask, L=L)
          xhat_fits = np.append(xhat_fits, {'xhat_fit': xhat_fit, 'obs_ids': splits[it]})
          #print(b_data[:4]); print(xhat_0[:4]); print(b_mask[:4]); print(mask[:4])
          b_xhat = xhat[splits[it],:]
          #b_xhat[batches_mask[it]] = np.mean(params_x['mean'].reshape([niw,-1]),axis=0).reshape([n,p])[splits[it],:][batches_mask[it]]   #  .cpu().data.numpy()[batches_mask[it]]  # keep observed data as truth
          b_xhat[~batches_mask[it]] = xhat_fit['xm'].cpu().data.numpy()[~batches_mask[it]] # just missing imputed
          xhat[splits[it],:] = b_xhat
        time_impute=np.append(time_impute,time.time()-t0_impute)
        err = mse(xhat,xfull,mask)
        mse_train_miss = np.append(mse_train_miss,np.array([err['miss']]),axis=0)
        mse_train_obs = np.append(mse_train_obs,np.array([err['obs']]),axis=0)
        
        zgivenx_flat = xhat_fit['zgivenx_flat'].cpu().data.numpy()   # L samples*batch_size x d (d: latent dimension)
        imp_weights=xhat_fit['imp_weights'].cpu().data.numpy()
        
        print('Observed MSE  %g' %err['obs'])     # observed values are not imputed
        print('Missing MSE  %g' %err['miss'])
        print('-----')
    saved_model={'encoder': encoder, 'decoder': decoder}
    mse_train={'miss':mse_train_miss,'obs':mse_train_obs}
    train_params = {'h':h, 'sigma':sigma, 'bs':bs, 'n_epochs':n_epochs, 'lr':lr, 'niw':niw, 'dim_z':dim_z, 'L':L, 'dec_distrib':dec_distrib, 'n_hidden_layers': n_hidden_layers}
    return {'train_params':train_params,'loss_fits':loss_fits,'xhat_fits':xhat_fits,'saved_model': saved_model,'zgivenx_flat': zgivenx_flat,'MIWAE_LB_epoch': MIWAE_LB_epoch,'time_train': time_train,'time_impute': time_impute,'imp_weights': imp_weights,'MSE': mse_train, 'xhat': xhat, 'mask': mask, 'norm_means':norm_means, 'norm_sds':norm_sds}
  else:
    # validating (hyperparameter values) or testing
    encoder=saved_model['encoder']
    decoder=saved_model['decoder']
    for ep in range(1,n_epochs):
      #perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      #batches_data = np.array_split(xhat_0[perm,], n/bs)
      #batches_mask = np.array_split(mask[perm,], n/bs)
      #for it in range(len(batches_data)):
      #  optimizer.zero_grad()
      #  encoder.zero_grad()
      #  decoder_x.zero_grad()
      #  decoder_r.zero_grad()
      #  b_data = torch.from_numpy(batches_data[it]).float().cuda()
      #  b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
      #  loss = miwae_loss(iota_x = b_data,mask = b_mask)
      #  loss.backward()
      #  optimizer.step()
      #time_train=np.append(time_train,time.time()-t0_train)
      perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      batches_data = np.array_split(xhat_0[perm,], n/bs)
      batches_mask = np.array_split(mask0[perm,], n/bs)
      splits = np.array_split(perm,n/bs)
      batches_loss = []
      encoder.zero_grad(); decoder.zero_grad()
      
      # loss_fit=miwae_loss(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda())
      # MIWAE_LB=(-np.log(K)-loss_fit['neg_bound'].cpu().data.numpy())
      # print('Epoch %g' %ep)
      # print('MIWAE likelihood bound  %g' %MIWAE_LB) # Gradient step      
      
      loss_fits = []
      for it in range(len(batches_data)):
        b_data = torch.from_numpy(batches_data[it]).float().cuda()
        b_mask = torch.from_numpy(batches_mask[it]).float().cuda()

        loss_fit = miwae_loss(iota_x = b_data, mask = b_mask)
        loss = loss_fit['neg_bound']
        batches_loss = np.append(batches_loss, loss.cpu().data.numpy())
        
        loss_fit.pop("neg_bound")
        loss_fits = np.append(loss_fits, {'loss_fit': loss_fit, 'obs_ids': splits[it]})
       
      total_loss = -np.sum(batches_loss)   # negative of the total loss (summed over K & bs)
      MIWAE_LB = total_loss / (K*n) - np.log(K)
        
      ### Now we do the imputation
      # xhat_fit=miwae_impute(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda(),L=L)
      # time_impute=np.append(time_impute,time.time()-t0_impute)
      # xhat[~mask] = xhat_fit['xm'].cpu().data.numpy()[~mask]
      # #xhat = xhat_fit['xm'].cpu().data.numpy()    # observed values are not imputed
      t0_impute=time.time()
      batches_data = np.array_split(xhat_0, n/impute_bs)
      batches_mask = np.array_split(mask0, n/impute_bs)
      splits = np.array_split(range(n),n/impute_bs)
      xhat_fits = []
      for it in range(len(batches_data)):
        b_data = torch.from_numpy(batches_data[it]).float().cuda()
        b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
        xhat_fit=miwae_impute(iota_x = b_data, mask = b_mask, L=L)
        xhat_fits = np.append(xhat_fits, {'xhat_fit': xhat_fit, 'obs_ids': splits[it]})
        b_xhat = xhat[splits[it],:]
        #b_xhat[batches_mask[it]] = torch.mean(loss_fit['params_x']['mean'].reshape([niw,-1]),axis=0).reshape([n,p])[splits[it],:].cpu().data.numpy()[batches_mask[it]]  # keep observed data as truth
        b_xhat[~batches_mask[it]] = xhat_fit['xm'].cpu().data.numpy()[~batches_mask[it]] # just missing imputed
        xhat[splits[it],:] = b_xhat
      #xhat_fit=nimiwae_impute(iota_xfull = cuda_xfull, iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda(),covar_miss = torch_covars_miss,L=L,temp=temp_min)
      time_impute=np.append(time_impute,time.time()-t0_impute)

      err = mse(xhat,xfull,mask)
      mse_train_miss = np.append(mse_train_miss,np.array([err['miss']]),axis=0)
      mse_train_obs = np.append(mse_train_obs,np.array([err['obs']]),axis=0)
      zgivenx_flat = xhat_fit['zgivenx_flat'].cpu().data.numpy()   # L samples*batch_size x d (d: latent dimension)
      imp_weights = xhat_fit['imp_weights'].cpu().data.numpy()
      print('Observed MSE  %g' %err['obs'])   # observed values are not imputed
      print('Missing MSE  %g' %err['miss'])
      print('-----')
    mse_test={'miss':err['miss'],'obs':err['obs']}
    saved_model={'encoder': encoder, 'decoder': decoder}
    return {'loss_fits':loss_fits,'xhat_fits':xhat_fits,'zgivenx_flat': zgivenx_flat,'saved_model': saved_model,'LB': MIWAE_LB,'time_impute': time_impute,'imp_weights': imp_weights,'MSE': mse_test, 'xhat': xhat, 'xfull': xfull, 'mask': mask, 'norm_means':norm_means, 'norm_sds':norm_sds}  
