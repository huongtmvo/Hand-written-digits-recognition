# Build Deep Learning Models to Recognize Handwritten Digits
The goal of the project is to experiment with state of the art machine learning techniques and to build the high accuracy model on test data. We also create a script to run model inference on a given image. 



## Project Objectives 
- Explore MNIST dataset of handwritten digits from https://yann.lecun.com/exdb/mnist/ 
- Build MLP, CNN models using Keras and fine-tune models to get high accuracy
- Test models on real world examples

## Results
- MLP solution: Accuracy is 98.15% on test set.
- CNN solution: Accuracy is 99.15% on test set. 
- Models perform well on real world examples 

## Directory Structure 
- `exp`: notebook to run various experements, place to explore data, improve model performance 
- `src`: final scripts to train model and run model inference 
- `images`: real world images example
- `models`: checkpoint of model that has been trained and saved using keras `model.save(...)`


**How to train model**

```
python3 src/train_mlp.py
python3 src/train_cnn.py
```

**How to run model inference in real-world image**

Image: 

![Alt text](images/2.png?raw=true "Real-world example")

```
$ python3 src/run_model.py mlp images/2.png

out = [[1.1773597e-27 0.0000000e+00 1.0000000e+00 6.6215482e-19 0.0000000e+00
  2.3951275e-14 4.9028755e-13 2.0657011e-11 1.4949686e-34 0.0000000e+00]]

Model predicted output = 2

```




 

