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

## Additional Information

- This repository attempts to replicate the architecture only. Performance may vary based on parameters implemented. Can change the same and experiment using the   ```train.py```  module.
- A sample usage of this can be found in the [Notebook here.](https://github.com/RohanMathur17/SqueezeNet-Implementation/blob/main/SqueezeNet_Implementation.ipynb)
