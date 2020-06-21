# Style, Genre and Artist Classification of Artwork

Increase in digitalization of artwork illustrates the importance of
classification of paintings based on artists, styles and the genre of paintings. Classification methodologies will indeed help the visitors as well as the curators in analyzing and visualizing the paintings in any museum at their own pace. Moreover, finding the
artist of a painting is a difficult task because most artworks of an artist may have a exclusive painting style and multiple artists can have same styles of paintings.

## Models
I have experimented with four models -
- Implementation of Vanilla Convolution Neural Network
- Statistical machine learning based approach using Random Forest Classifier by incorporating Visual Bag of Words Technique.
- Implementation of Capsule Networks
- Transfer Learning using pretrained Networks like AlexNet

## Dataset

The dataset for this classification task is Wikiart dataset obtained from reference [Wikiart](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset). The following figure illustrates the distribution of training vs testing images for the dataset.

![Image](https://github.com/Laalasa137/Classification-of-Artwork/blob/master/Figures/a.png)

## Model 1 - CNN Implementation

Convolutional Neural Networks are a variation of this simple feed forward network possessing a robust architecture. The image vectors are given as an input to the CNN - which then computes the weights, loss function to obtain the final output and identify corresponding classes. CNN is standard architecture which can be worked with images, to capture features corresponding to images.

![Image](https://github.com/Laalasa137/Classification-of-Artwork/blob/master/Figures/cnnarch.png)

## Model 2 - Random Forest Classifier by incorporating Visual Bag of words

This is a successful image classification technique described in paper [Image classification based on support vector machine and the fusion of complementary features](https://arxiv.org/ftp/arxiv/papers/1511/1511.01706.pdf)

The following flowchart indicates the step by step implementation of the technique used -
![Image](https://github.com/Laalasa137/Classification-of-Artwork/blob/master/Figures/RF_implementation.PNG)

## Model 3 - Implementation of capsule networks

[Capsule Networks](https://arxiv.org/pdf/1710.09829.pdf) is a robust architecture to capture spatial relationships in an image unlike CNN's. Explored Capsule Networks for this dataset by using base code forked from [handson-ml](https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb) and modified the code accordingly.


Architecture - 
![Image](https://github.com/Laalasa137/Classification-of-Artwork/blob/master/Figures/Capsule_arch.PNG)

## Model 4 - Transfer Learning with AlexNet

[Transfer learning](https://www.mathworks.com/help/deeplearning/ug/transfer-learning-using-alexnet.html) is commonly used deep learning approach where pre-trained networks
are used to capture rich feature representations. [AlexNet](https://pdfs.semanticscholar.org/d02d/7d5edbae49cdce5da934faa41dd383633a22.pdf?_ga=2.119643855.61356801.1592779147-1953199154.1573935763) is CNN with a 8 layers deep
architecture.

![Image](https://github.com/Laalasa137/Classification-of-Artwork/blob/master/Figures/alex_arch.png)

## Installation and running the code

- Install Anaconda, then install dependencies listed in the ```dependencies.txt``` file.

- All the folders in wikiart folder contains images of varied dimensions which are all different paintings. The wikiart csv folder contains the csv files which denote the path of the image and the class it belongs to.
Run ```python datapreprocessing.py``` to convert all the images into same size(227X227) and carefully categorize the images into train, test and validation folders respectively for each of artist, genre and style classification tasks.


- Model1(Implemented using tensorflow) -> Run ```python finalCNN.py```
- Model2(Implemented with python) -> Run ```python rftraining.py``` and then run ```python rftesting.py```
- Model3(Implemented using tensorflow) -> Run ```python capsuleNet.py```
- Model4(Implemented using MATLAB) -> Run the file ```transferlearning.m```

### Performance & Results

- For Classification based on artists

| Method  | Accuracy |
| ------------- | ------------- |
| Random Forest Implementation  | 27.12  |
| CNN Implementation  | 43.39  |
| Capsule Net Implementation  | 17.18  |
| Transfer Learning using AlexNet  | **67.30** |

- For Classification based on style

| Method  | Accuracy |
| ------------- | ------------- |
| Random Forest Implementation  | 10.18  |
| CNN Implementation  | 42.48  |
| Capsule Net Implementation  | 18.75  |
| Transfer Learning using AlexNet  | **49.37** |

- For Classification based on genre

| Method  | Accuracy | 
| ------------- | ------------- |
| Random Forest Implementation  | 25.14  |
| CNN Implementation  | 53.42  |
| Capsule Net Implementation  | 19.49  |
| Transfer Learning using AlexNet  | **65.01** |

Performance with Transfer Learning architecture is better than the other models explored. 

## Visualizations

SIFT Features Visualization, TSNE Visualizations, Activations of the first layer for AlexNet Model, Training vs Test plots for each of the models explored is included in the code base.

Receiver operating characteristic(ROC) Curve for artist based classification - 

![Image](https://github.com/Laalasa137/Classification-of-Artwork/blob/master/Figures/artist_ROC.PNG)

## References
1. [Wikiart Dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt)
2. [Image Classification using Random Forests and Ferns](https://www.cse.huji.ac.il/~daphna/course/CoursePapers/bosch07a.pdf)
3. [Capsule Networks](https://arxiv.org/pdf/1710.09829.pdf)
4. [Image classification based on support vector machine and the fusion of complementary features](https://arxiv.org/ftp/arxiv/papers/1511/1511.01706.pdf)
5. [Transfer learning](https://www.mathworks.com/help/deeplearning/ug/transfer-learning-using-alexnet.html)
6. [Image Classification in python using visual bag of words](https://ianlondon.github.io/blog/how-to-sift-opencv/)

