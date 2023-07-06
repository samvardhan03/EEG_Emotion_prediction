# EEG_Emotion_prediction

EEG emotion detection is a technique that uses electroencephalography (EEG) to measure brain activity and identify different emotional states. EEG is a non-invasive method that measures electrical activity in the brain through electrodes placed on the scalp. The electrical activity of the brain is related to different cognitive and emotional processes, and EEG can be used to measure changes in brain activity that occur in response to different emotions.

#About the dataset:

The dataset you provided is a dataset of EEG signals that have been labeled with the corresponding emotional state. The dataset contains 1000 trials, each of which is a 4 second EEG signal. The emotional states that are represented in the dataset are happiness, sadness, anger, and fear. features in the dataset: Time domain features:
Mean: The average value of the EEG signal over time.
Standard deviation: The standard deviation of the EEG signal over time.
Peak-to-peak amplitude: The difference between the maximum and minimum values of the EEG signal over time.
Root mean square: The square root of the mean of the squared values of the EEG signal over time.
Frequency domain features:

Power spectral density (PSD): The power of the EEG signal as a function of frequency.

Mean frequency: The average frequency of the EEG signal.

Dominant frequency: The frequency of the EEG signal with the highest power. Time-frequency domain features:

Hjorth parameters: A set of three parameters that characterize the shape of the EEG signal's power spectrum.

Wavelet transform: A mathematical transform that decomposes the EEG signal into a series of wavelets.

Short-time Fourier transform: A mathematical transform that decomposes the EEG signal into a series of frequency components over a short time window. Spatial domain features:

Laplacian of the EEG signal: A measure of the spatial variation of the EEG signal.

Coherence between different electrodes: A measure of the correlation between the EEG signals at different electrodes.

3.  • selecting all of the columns in the data DataFrame that start with fft_. This will select the frequency domain features of the EEG signals in the dataset. The frequency domain features are calculated from the power spectrum of the EEG signal and represent the distribution of power across different frequencies. The power spectrum is a graph that shows the amount of power in the EEG signal at different frequencies. The frequency domain features can be used to identify changes in brain activity that are associated with different emotions.

The full form of fft_0_b and fft_749_b is Fast Fourier Transform (FFT) 0 Hz to 749 Hz. FFT is a mathematical algorithm that can be used to decompose a signal into its constituent frequencies. The fft_0_b and fft_749_b columns in the DataFrame likely contain the frequency data for the first 750 frequencies in the signal.

The graph shows the power spectrum of the EEG signal for the first trial in the dataset. The x-axis shows the frequency of the EEG signal, and the y-axis shows the power of the EEG signal at each frequency. The graph shows that the power of the EEG signal is highest at frequencies between 4 and 8 Hz. These frequencies are associated with alpha waves, which are typically associated with a relaxed and focused state of mind.

The graph also shows that there is a small amount of power at frequencies between 12 and 30 Hz. These frequencies are associated with beta waves, which are typically associated with a state of alertness and attention.

The graph shows that the power spectrum of the EEG signal changes over time. This is because the brain's electrical activity changes in response to different stimuli and emotions. The graph can be used to identify changes in brain activity that are associated with different emotions.

The code above provided reshapes the training and testing data sets from 2D to 3D arrays. This is necessary for some machine learning models, such as convolutional neural networks (CNNs).
A 2D array is a matrix with two dimensions: rows and columns. A 3D array is a matrix with three dimensions: rows, columns, and channels. The channels dimension is used to represent different features of the data. For example, in an image dataset, the channels dimension might represent the red, green, and blue channels of the image.
The code above provided reshapes the training and testing data sets by adding a new dimension to the end of the array. The new dimension has a size of 1, which represents the number of channels. This ensures that the data sets are in the correct format for use with CNNs.
network architecture :
The Flatten and Dense layers in Keras are used to build deep learning models. The Flatten layer is used to convert the input data into a one-dimensional vector, while the Dense layer is used to create a fully connected layer.

The Flatten layer is used when the input data is in a multidimensional format, such as a 2D image or a 3D tensor. The Flatten layer converts the input data into a one-dimensional vector, which can then be used by the Dense layer.

The Dense layer is a fully connected layer, which means that each neuron in the layer is connected to every neuron in the previous layer. The Dense layer is used to learn the relationships between the input data and the output data.

The code above provided creates a recurrent neural network (RNN) model using the TensorFlow Keras library. The model has an input layer with shape (X_train.shape[1], 1), a GRU layer with 512 units, a Flatten layer, and a Dense layer with 3 units and a softmax activation function.

The input layer receives the input data. The GRU layer is a recurrent layer that uses a gated recurrent unit (GRU) to learn long-term dependencies in the data. The Flatten layer converts the output of the GRU layer into a one-dimensional vector. The Dense layer with 3 units and a softmax activation function is used to make predictions.

The model is then summarized using the model.summary() method. The summary shows the number of parameters in the model, the shape of the input and output tensors, and the activation functions used in each layer.

GRU stands for gated recurrent unit. It is a type of recurrent neural network (RNN) that is used to learn long-term dependencies in sequential data. GRUs are similar to LSTMs (long short-term memory), but they have fewer parameters and are therefore easier to train.

GRUs work by using two gates to control the flow of information through the network: the update gate and the reset gate. The update gate decides how much of the previous hidden state to keep, while the reset gate decides how much of the previous hidden state to forget. The output of the GRU is then a linear combination of the current input and the updated hidden state.

GRUs have been shown to be effective for a variety of tasks, including natural language processing, speech recognition, and machine translation. They are a popular choice for these tasks because they can learn long-term dependencies in sequential data, which is often important for these tasks.

Training:  • The loss function used will be 'Categorical_CrossEntropy'. We will be using callback functions like Early_Stopping to avoid overfitting and lr_scheduler to change the learning rate while model trains.

The EarlyStopping callback stops training when the model stops improving on a validation set. This can prevent overfitting, which is a problem that occurs when a model learns the training data too well and is not able to generalize to new data.
The ModelCheckpoint callback saves the model weights to a file at regular intervals. This can be used to save the best model weights, which can then be used for inference or further training.
