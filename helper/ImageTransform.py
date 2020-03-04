import numpy as np 
import pywt
import cv2
RESIZE_SHAPE = (384,256) # Width x Height 

def load_preprocess(image_path,mode="data"):

  image = cv2.imread(image_path)
  
  #image = np.arrayimage)
  num_dim = len(image.shape)
  if mode == "data":
    image = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    image = np.transpose(image,(1,0,2))

  elif mode =="label":
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = image.T

  image = cv2.resize(image, dsize=RESIZE_SHAPE, interpolation=cv2.INTER_CUBIC)
  return image

def img2batch(arr,shape=(32,32),strides=(32,32)):

    nrows, ncols = shape
    batches = []
    dim = len(arr.shape)
    if dim == 3:
        h, w,_ = arr.shape
    elif dim ==2:
        h, w = arr.shape
        
    stride_h, stride_w = strides

    for r in range((h-nrows)//stride_h+2):
        
        # Get row start and destination
        from_r = r*stride_h
        to_r = from_r +  nrows
        
        # If out-of-bound rows, break
        if to_r > h:
            break
            
        for c in range((w-ncols)//stride_w+1):
            
            # Get column start and destination
            from_c = c*stride_w
            to_c = from_c + ncols
            
#             print("r {} to {}, c {} to {}".format(from_r,to_r,
#                                                   from_c,to_c))

            # If out-of-bound columns, break
   
            if to_c > w:
                break
                
            if dim == 3:
                batches.append(arr[from_r:to_r,from_c:to_c,:])
            elif dim==2:
                batches.append(arr[from_r:to_r,from_c:to_c])

    return batches
  
def mu_std_sum(matrix):
  return [np.mean(matrix),np.std(matrix),np.sum(matrix)]

def extract_features(image):

  # Transpose to iterate through channels
  image = np.transpose(image,(2,0,1))
  vector = []

  for channel in image[1:]: # Remove Y Channel

    # iterate through db1->db5
    for i in range(1,6):
      coeffs = pywt.wavedec2(channel,"db"+str(i),level=3)

      # Append first mat mu_std_sum
      vector += mu_std_sum(coeffs[0])

      for dec in coeffs[1:]:
        vector += mu_std_sum(dec[0])
        vector += mu_std_sum(dec[1])
        vector += mu_std_sum(dec[2])

  return vector
      
