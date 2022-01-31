# SqueezeNet-Implementation

This repository attempts to replicate the SqueezeNet architecture using TensorFlow discussed in the research paper: "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size".

The paper can be read [here.](https://arxiv.org/pdf/1602.07360.pdf)

The official implementation of this paper can be found [here.](https://github.com/forresti/SqueezeNet)

## Requirements

```
1. tensorflow-gpu==1.13.1 ( Does not work with Tensorflow 2.x)
2. sklearn
3. opencv-python
4. numpy
5. Python 3.x ( Specifically not python 3.8, anything else works)
```

## Architecture Implemented

1. Fire Module

<img src="https://miro.medium.com/max/875/1*dVaL1bcv5Ewpz-wen7IXCA.png" width="700">


2. SqueezeNet Module

<img src="https://pytorch.org/assets/images/squeezenet.png" width="700">

## Working

The data used for this implementation was picked up from the Kaggle Dataset - [Soil Types](https://www.kaggle.com/prasanshasatpathy/soil-types)

- Step 1: Clone the repository
```
git clone https://github.com/RohanMathur17/SqueezeNet-Implementation.git
```

- Step 2: Install necessary libraries as discussed in Requirements section
- Step 3: Within train.py, change your path for data at line 31
```
Change this line 
base_dir = '/content/gdrive/MyDrive/SqueezeNet/data/'
```

- Step 4: In your command prompt, run the train.py file to train the model
```
python train.py
```

## Results

The model was trained on the Soil Data, which contained five categories of soil with around 30-35 images in each category. The dataset was split into 75% training data, 25% test data. Model training parameters were as follows - 
```
- Optimizer: Adam
- Learning Rate: 0.001
- No. of Epochs: 500
- Metrics: Categorical_Accuracy
```
Final figures for ```training_loss, training_categorical_accuracy, validation_loss, validation_categorical_accuracy ``` are given in the following image-

<img src="https://github.com/RohanMathur17/SqueezeNet-Implementation/blob/main/images/results.png" width="700">


## Additional Information

- This repository attempts to replicate the architecture only. Performance may vary based on parameters implemented. Can change the same and experiment using the   ```train.py```  module.
- A sample usage of this can be found in the [Notebook here.](https://github.com/RohanMathur17/SqueezeNet-Implementation/blob/main/SqueezeNet_Implementation.ipynb)
