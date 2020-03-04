import numpy as np


class MultiRegressor:
  def createModel(self,output_size):
    regressor = Sequential()
    regressor.add(Dense(output_size))
    regressor.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])
    return regressor
    
  def __init__(self,output_size,num_regressors=5):
    self.output_size = output_size
    self.num_regressors = num_regressors
    self.regressors_bank = []
    self.scaler_obj = MinMaxScaler()
    self.initialized_mat = None
    self.regressors_weight = []
    self.output_size = output_size
    self.mode = "train" 
    # To store weight if loaded pretrained model
    self.W = []
    self.b = []
    # "train" for model train from scratch
    # "pretrained" for load pretrained model
    
      
  def predict(self,X):
    num_data = X.shape[0]

    last_init_mat = None
    last_predict = None
    last_shifting_mat = None
    initialized_mat = None
    shifting_pred = None
    # to cache X,Y for last model
    X_ = None

    if self.mode == "train":  
      for idx,regressor in enumerate(self.regressors_bank):

        # if its the first regressors then random initialized
        if idx == 0:
          # initialize landmark by the mean of all landmarks
          initialized_vector= self.initialized_vector
          
          # Give initialized prediction for landmarks
          initialized_mat = np.ones((num_data,self.output_size))*initialized_vector

        else:
          # Get the last model
          # Get the last model
          last_model = self.regressors_bank[idx-1]

          last_shifting_pred = last_model.predict(X_)

          # this is shifting pred from last regressor
          initialized_mat = initialized_mat + shifting_pred 
        
        # Scale the initialized point
        if self.mode == "train":
          initialized_mat_scaled = self.scaler_obj.transform(initialized_mat)
        elif self.mode =="pretrained":
          initialized_mat_scaled = (initialized_mat - self.data_min)/(self.data_max-self.data_min)

        # Get input data to k-th regressor
        X_ = np.concatenate((X,initialized_mat_scaled),axis=1)
        shifting_pred = regressor.predict(X_)
        
    elif self.mode == "pretrained":
      for idx in range(self.num_regressors):

        # if its the first regressors then random initialized
        if idx == 0:
          # initialize landmark by the mean of all landmarks
          initialized_vector= self.initialized_vector
          
          # Give initialized prediction for landmarks
          initialized_mat = np.ones((num_data,self.output_size))*initialized_vector

        else:
          # Get the last model
          # Get the last model
          # last_model = self.regressors_bank[idx-1]

          # last_shifting_pred = last_model.predict(X_)

          # this is shifting pred from last regressor
          initialized_mat = initialized_mat + shifting_pred 
        
        # Scale the initialized point
        if self.mode == "train":
          initialized_mat_scaled = self.scaler_obj.transform(initialized_mat)
        elif self.mode =="pretrained":
          initialized_mat_scaled = (initialized_mat - self.data_min)/(self.data_max-self.data_min)

        # Get model weight
        W = self.W[idx]
        b = np.expand_dims(self.b[idx],0)
        
        # Get input data to k-th regressor
        X_ = np.concatenate((X,initialized_mat_scaled),axis=1)
        shifting_pred = X_.dot(W)+ b
        

    return initialized_mat+shifting_pred

  def fit(self,X,Y,validation_split=0.2,max_epochs=20):
    for i in range(self.num_regressors):
      new_model = self.createModel(self.output_size)
      self.regressors_bank.append(new_model)

    num_data = X.shape[0]

    # Fit the scaler
    self.scaler_obj.fit(Y)

    last_init_mat = None
    last_predict = None

    # to cache X,Y for last model
    X_ = None
    Y_ = None
    
    ES_callback = EarlyStopping(monitor='val_mean_absolute_error',patience=3,restore_best_weights=True,verbose=1)


    for idx,regressor in enumerate(self.regressors_bank):

      # if its the first regressors then random initialized
      if idx == 0:
        # initialize landmark by the mean of all landmarks
        initialized_vector= np.expand_dims(Y.mean(axis=0),axis=0)

        # Give initialized prediction for landmarks
        initialized_mat = np.ones((num_data,self.output_size))*initialized_vector
        self.initialized_vector = initialized_vector.copy()
        last_init_mat = initialized_mat

      else:
        # Get the last model
        last_model = self.regressors_bank[idx-1]

        last_shifting_pred = last_model.predict(X_)

        initialized_mat = initialized_mat + last_shifting_pred

      shifting_mat = Y - initialized_mat
      initialized_mat_scaled = self.scaler_obj.transform(initialized_mat)

      # Training and target data for current regressor
      X_ = np.concatenate((X,initialized_mat_scaled),axis=1)
      Y_ = shifting_mat
      
      regressor.fit(X_,Y_,epochs=max_epochs,callbacks=[ES_callback],validation_split=validation_split)
  
  def save_regressors(self,path):

    # Save min max for scaler
    data_min = np.expand_dims(MR.scaler_obj.data_min_,0)
    data_max = np.expand_dims(MR.scaler_obj.data_max_,0)
    np.save(path + "data_min.npy",data_min)
    np.save(path + "data_max.npy",data_max)

    # Save initialized mat
    np.save(path + "init_vector.npy",self.initialized_vector)

    # Save weights
    for idx,regressor in enumerate(self.regressors_bank):
      W = regressor.layers[0].get_weights()[0]
      b = regressor.layers[0].get_weights()[1]
      np.save(path + "W" + str(idx) + ".npy",W)
      np.save(path + "b" + str(idx) + ".npy",b)
    print("model saved to " + path)

  def load_regressors(self,path):
    # Set mode to "pretrained"
    self.mode = "pretrained"

    # Load pretrained scaler attribute
    self.data_min = np.load(path + "data_min.npy")
    self.data_max = np.load(path + "data_max.npy")

    # Save initialized mat
    self.initialized_vector = np.load(path + "init_vector.npy")

    for idx in range (self.num_regressors):
      self.W.append(np.load(path + "W" +str(idx)+".npy"))
      self.b.append(np.load(path + "b" +str(idx)+".npy"))
    print("model loaded from" + path)
