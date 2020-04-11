# Training Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_loader():
    
    # Load dataset file
    data_frame = pd.read_csv(r'..\data\Detection_Data\FacialKeyPoints\training.csv')
    
    data_frame['Image'] = data_frame['Image'].apply(lambda i: np.fromstring(i, sep=' '))
    data_frame = data_frame.dropna()  # Get only the data with 15 keypoints
   
    # Extract Images pixel values
    imgs_array = np.vstack(data_frame['Image'].values)/ 255.0
    imgs_array = imgs_array.astype(np.float32)    # Normalize, target values to (0, 1)
    imgs_array = imgs_array.reshape(-1, 96, 96, 1)
        
    # Extract labels (key point cords)
    labels_array = data_frame[data_frame.columns[:-1]].values
    labels_array = (labels_array - 48) / 48    # Normalize, traget cordinates to (-1, 1)
    labels_array = labels_array.astype(np.float32) 
    
    # shuffle the train data
    # imgs_array, labels_array = shuffle(imgs_array, labels_array, random_state=9)  
    
    return imgs_array, labels_array




if __name__ == "__main__":

    imgs, labels = data_loader()
    print(imgs.shape)
    print(labels.shape)

    n=0
    labels[n] = (labels[n]*48)+48
    image = np.squeeze(imgs[n])
    plt.imshow(image, cmap='gray')
    plt.plot(labels[n][::2], labels[n][1::2], 'ro')
    plt.show()