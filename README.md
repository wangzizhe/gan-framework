# Software Framework for Image Data Synthesizing based on GAN for ML-Systems in Digital Health



## Aim

On the one hand, **lack of data** is a common problem in the field of Digital Health, on the other hand, medical imaging always require **high definition** images. The aim of this framework is **generalizing the process** to make new images using medical images with GAN and **make this framework easily to be configurated** for different use cases with different custom datasets and different GANs.



## Structure of the framework

As shown in this picture, this framework consists of 4 parts, each part is a modular structure, it can be easily changed or modified for different use cases. 

![](C:\Users\12081\Desktop\GAN\DA\06_Software_Framework\Framework.png)

### Part 1:

With part 1 it is easily to **choose different GANs** and **create proper environments** so that this GAN can work smoothly. 

### Part 2:

With part 2 it is easily to use **different datasets** and **prepare the data** for training the GAN.

### Part 3:

With part 3 it will first show which **hyperparameters** are needed. This part provides an **automated hyperparameter tuning** function. After training you can choose the best model to **generate new images**.

### Part 4:

After new images are generated, you can **couple a CNN Classifier** or other tools / systems to check whether these generated image are "true enough".



---



Basic idea is, you have different repositories of different GAN models, if you have chosen one GAN model, you can enter this model's repository and use this GAN model directly. 

In part 1 and part 3, because the environment `environment.xml`, hyperparameters which are used `show_hyperparameters.py`, how to train, automatically hyperparameter optimize `optimize.py` and generate `generate.py` depend on the model, for each model  there are those 4 files in their repositories.

In part 2, because it is faster to read one file than read many files in a directory, modern models can read a `.zip` file as input, the `data_preparation.py` provides a general approach to transfer images to a `.zip` file, and the image sizes can be changed with this script. 

In part 4, this part can be seen as a whole part, different tools / methods can be coupled to here, for my purpose, a CNN Classifier is coupled after GAN, so that mixed (real or generated) images after data preparation with`data_preparation_classifier.py` can be sent to the CNN Classifier to check the result.  



## To-dos

According to the plan, to finish the pipeline there are 4 To-dos which need to be done.

1.  A script `environment.xml` which saves the environment this model uses and after running this script a virtual environment will be created so the model can run smoothly in this virtual environment.
2. A script `hyperparameters.py` which stores information of all the hyperparameters, after running this script, people will know what kind of hyperparameters they should type in and what are the meanings of these hyperparameters.
3. A script `optimize.py` which can run training and hyperparameter optimization automatically using some tools for instance [Optuna](https://optuna.org/) or [NNI](https://nni.readthedocs.io/en/stable/).
4. A script `data_preparation_classifier.py` which prepares data for the CNN classifier, e.g. the script can realize how many % real data and how many % generated data will be used for the Classifier.![](C:\Users\12081\Desktop\GAN\DA\06_Software_Framework\GAN_Pipeline.jpg)