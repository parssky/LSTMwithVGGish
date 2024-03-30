---

Audio Processing with LSTM & VGGish (example story)

Introduction
In English proficiency tests, there is an important section called Speaking where the test-taker must read a paragraph aloud, following proper pronunciation, accent, word accuracy, clarity, speed, fluency, and the appropriate tone (declarative, exclamatory, or interrogative) for each sentence. Ultimately, this reading performance is scored, and this score determines whether the test-taker passes or fails the exam.
However, the main issue lies in the fact that this test is evaluated qualitatively by judges and lacks a precise formula or framework. In this report, I aim to develop a quantitative solution for this test.
The primary challenge of this project is to create a criterion that can estimate the probable score of the test-taker. Additionally, it is challenging to quantify the aforementioned features. Another challenge is that each test-taker reads different paragraphs, meaning there is no fixed, identical sentence being read. Consequently, the recorded audio from the test-takers varies in length, and general factors such as gender, age, place of origin, etc., can influence the quality of the performance, further complicating the matter.
Proposed Methodology:

Utilizing artificial intelligence methods is a highly suitable option for addressing this challenge. This is because, without considering the individual preferences of the judges, we employ an AI model for evaluation, which can estimate scores from our data.
Comparison of AI Methods:
Neural Networks: In this method, hardware challenges are apparent. Additionally, working with neural networks is considerably more challenging compared to machine learning algorithms. The main feature of this method is automatic feature extraction and the ability to use pre-trained networks.
Classic Machine Learning: This method does not require very powerful hardware and implementation is much easier with libraries like Scikit-Learn. However, feature extraction needs to be done manually.
In this research, considering the numerous features of sound and the availability of pre-trained networks for audio, deep neural networks have been employed.
Feature Extraction:
For feature extraction, we focused on speech processing methods.
Mel-Frequency Cepstral Coefficients (MFCC) are a common method for extracting features from audio signals, typically used in speech processing applications such as speech recognition, speaker identification, and sound classification.
MFCCs bear a strong resemblance to the human ear and are considered a method for simulating the human auditory system. The MFCC extraction process involves the following stages:
Windowing: The audio signal is divided into smaller time intervals (usually 20–40 milliseconds).
Fourier Transform: In each window, a Fourier transform is taken to obtain the frequency spectrum of the signal.
Mel Filtering: The frequency spectrum is weighted by Mel filter banks that simulate the human ear.
Logarithmic Scaling: Logarithmic scaling is applied for non-linear compression to simulate human auditory sensitivity.
Discrete Cosine Transform (DCT): Applied to obtain spectral coefficients or MFCCs. MFCCs represent the shape of the spectrum.
Usually, a number of the first MFCC coefficients, along with features such as energy and numerical averages, are used to describe a complete time interval of the signal.
MFCCs have the ability to capture human auditory characteristics and perform well in applications such as speech recognition, music classification, and environmental sound detection. They provide the possibility of reducing the dimensions of audio data and are often the first step in speech processing systems.
VGGish Network:
VGGish is a convolutional neural network (CNN) model trained by Google researchers for feature extraction from audio content. This model is based on the VGGNet architecture designed for image recognition, but adapted for feature extraction from audio signals such as speech, music, and environmental sounds.
The VGGish architecture consists of 8 convolutional layers and a linear mapping layer at the end. The convolutional layers consist of convolutional, pooling, and activation layers. Each convolutional layer has 3x3 convolution filters trained to extract various features from audio signals.
The feature extraction process with VGGish is as follows:
The input audio signal is divided into 960-millisecond time blocks.
Each block is converted into a 96x64 frequency-time spectrum suitable for input to the convolutional layers.
The convolutional layers and pooling layers sequentially extract features from the frequency-time spectrum.
The final linear mapping layer compresses the extracted features from previous layers into a 128-dimensional vector of audio features.
This 128-dimensional vector is a powerful representation of the input audio content that can be used for classification, sound event detection, and other audio processing applications. The VGGish model has been trained on a vast dataset of diverse audio content to extract meaningful features from speech, music, environmental sounds, etc.
It's important to note that VGGish is just a feature extraction model and needs to be combined with another machine learning model such as convolutional neural networks or recurrent neural networks for specific applications like classification or sound detection.
Combining VGGish and MFCC:
VGGish and MFCC are two different yet complementary approaches for feature extraction from audio signals. The relationship between them is as follows:
MFCC is a mathematical and signal-centric method for extracting basic features from audio signals. Inspired by the human auditory system, this method extracts features such as spectral coefficients, energy, etc., from audio signals. MFCCs are basic features that represent important patterns in audio signals.
On the other hand, VGGish is a deep learning and data-centric model for feature extraction from audio. This model, trained on a large dataset of diverse audio content, learns to extract more complex and meaningful features from audio.
The relationship between these two approaches is that MFCCs are usually used as the initial input to the VGGish model. That is, MFCCs are first extracted from the audio signal, and then these basic features are fed into the VGGish model to learn more complex and meaningful features from them.
This combination allows us to take advantage of the strengths of both approaches. MFCCs provide basic features related to the human auditory system, while VGGish transforms these basic features into deeper and more meaningful features that are more useful for specific audio processing tasks.
Therefore, in advanced audio processing systems, we have used these two approaches in combination to extract rich and efficient features from audio content.

Preprocessing:
A) The current dataset consists of 299 audio files in .mp3 format, which is used for compression. To extract the necessary features, we need to convert them to .wav format and extract them from the compressed state.
B) When extracting embeddings from the VGGish network, the dimensions of the embeddings are in the form [128*n], where n varies depending on the length of the audio. In the current dataset, the minimum length is 19 and the maximum is 37. After extracting MFCCs, our embedding matrix changes to [n*26], where the number 26 refers to the number of extracted features, and n varies depending on the length of the audio. When feeding these features into the deep neural network model, it's necessary for all dimensions to be equal. To achieve this, we use padding. We transform the dimensions of all audio files to match the largest audio file, and any missing data is filled with zeros (Zero Padding).
C) At this stage, we normalize all inputs and scores. The MIN-Max method is used, and our data range is scaled between 0 and 1.

Dataset:
Our dataset consists of two columns and 299 rows. The first column contains the name of the audio files, and the second column contains the corresponding scores.
Dataset Construction:
Following the mentioned steps, the final dataset consists of one column containing the extracted MFCCs from the embeddings, normalized, and the second column containing the normalized scores. Additionally, the current dataset is divided into two parts, 80% for training and 20% for evaluation.

Neural Network Model:
In this research, due to the nature of audio waves, a Sequential neural network model has been utilized. There are three common types of deep neural network models for this data:
RNN:
- They have internal loops that carry information from one time step to the next.
- Capable of learning long-term patterns in sequences.
- However, they suffer from the "vanishing gradient" problem in practice, making them struggle with learning long-term dependencies.
LSTM:
- They are a special kind of RNN designed to address the vanishing gradient problem.
- They have internal gates that decide what information to add, retain, or discard from the cell memory.
- These gates allow LSTMs to better learn long-term dependencies and avoid the vanishing gradient problem.
- However, the more complex architecture of LSTMs makes their computations heavier than simple RNNs.
In general, LSTMs are a better option than simple RNNs for most sequence data-related tasks because they can better learn long-term patterns. However, in cases where the sequence length is short or computational resources are limited, using simple RNNs may be more suitable.
In this research, due to the long duration of the audio files, LSTM networks have been used. Transformer networks have been set aside due to hardware limitations.
Training the Network:
Initially, the mean squared error (MSE) loss function is defined, and hyperparameter values are set. The network training begins. Techniques such as regularization and dropout are used to prevent overfitting.
Hyperparameters:
Number of neurons
- Number of hidden layers
- Learning rate
- Number of epochs
- Dropout rate
- Regularization rate
- Number of MFCC features

Results and Evaluation:
Based on the evaluation results, it appears that the third model has a lower error. However, the overall error on the test data should ideally be between 0.001 to 0.01 to be considered acceptable. Due to computational complexity, the LSTM model could not be trained with a larger and deeper network. Therefore, we have settled for these results.
Future Work:
To improve the model in the future, the following strategies can be considered:
1. Increasing the dataset size: Expanding the dataset volume for deep neural networks can potentially enhance performance.
2. Using more powerful hardware: Utilizing stronger hardware resources than the free Google Colab environment used in the current research can lead to better results.
3. Modifying the LSTM network: Changing the LSTM architecture sometimes improves results.
4. Altering feature extraction methods: Experimenting with different feature extraction techniques such as LPCCs, PLPs, Rasta-PLP, GFCCs, energy-based features, phase angle-based features, wavelet transform-based features, or features extracted from convolutional neural networks like VGGish might yield better results.
5. Changing the embedding extraction network: Switching to a different network architecture like VGGVox for embedding extraction could be beneficial.
6. Data formatting changes: Using data with uniform lengths can simplify processing and potentially improve model performance.

---

Uploading the dataset to GitHub is not possible due to its large size. You can email it to receive it
spacenavard1@gmail.com
---

