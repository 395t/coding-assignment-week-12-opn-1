# Open world perception

## Code structure

All of our code can be found under the `notebooks` folder in our repository.

## Commands to Reproduce Experiments

In order to run the notebooks we recommend the usage of Google Colab notebooks.

## Task

The task that we are studying is using different methods of contrastive learning to do image classification.

## Datasets

We evaluate our approach on 3 publically avaliable datasets.

#### CIFAR-10

This dataset contains 60k color images which are uniformly distributed accross 10 classes. The images are of size 4x32x32.
- Alex Krizhevsky. [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), 2009.

#### STL-10

This dataset contains 500 training images as well as one thousand testing images per class. Additionally, the dataset also contains 100k unlabeled images, which do not use for training, thus our model doesn't have state of the art performance.

- Adam Coates, Honglak Lee, Andrew Y. Ng. [An Analysis of Single Layer Networks in Unsupervised Feature Learning AISTATS](https://cs.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf), 2011

#### Caltech-101

This dataset consists of colour images of objects belonging to 101 classes.
- Fei-Fei, Li, Rob Fergus, and Pietro Perona. "[One-shot learning of object categories.](http://vision.stanford.edu/documents/Fei-FeiFergusPerona2006.pdf)" IEEE transactions on pattern analysis and machine intelligence 28.4 (2006): 594-611.

## Experiments

The topics relating to optimizers address different concerns and the hyperparameters of each approach are vastly different. Therefore, we have experiments for each idea and then some comparisons at the end.

# MoCo
We fine-tuned a pre-trained MoCo model on the three image classifiation dataset and compare it to a ResNet-50 model, randomly initialized and fine-tuned with the same data. The code is located in `notebooks/moco.ipynb`.

We downloaded the pre-trained MoCo model from [the MoCo github](https://github.com/facebookresearch/moco). The weights are for a ResNet-50, which We added a linear layer with the corresponding classes size for each dataset to. According to the MoCo paper they trained for 200 epochs over the [ImageNet-1M](https://www.image-net.org) and [Instagram-1B](https://paperswithcode.com/dataset/ig-1b-targeted), which took ~6 days. As comparison, We also implemented a ResNet-50 with the same layers, but randomly initialized.

Both models were trained using Adam optimizer, a learning rate of .0001, and cross entropy loss.

## CIFAR-10

![loss|50%](images/train_loss_cifar.png)

![train|50%](images/train_acc_cifar.png)

![test|50%](images/test_acc_cifar.png)

## Caltech-101

![loss|50%](images/train_loss_cal.png)

![train|50%](images/train_acc_cal.png)

![test|50%](images/test_acc_cal.png)

## STL-10

![loss|50%](images/train_loss_stl.png)

![train|50%](images/train_acc_stl.png)

![test|50%](images/test_acc_stl.png)

## Observations

| Datasets/Model | MoCo | ResNet-50 |
|---|---|---|
| CIFAR-10 | 87.42 | 73.72 |
| Caltech-101 | 25.08 | 18.05 |
| STL10 | 54.45 | 23.28 |

MoCo greatly outperforms ResNet-50 on all three datasets. The pre-training done for MoCo Seems not only speed up convergence but also increase classification accuracy. This is most likely because it creates important pathways during the unsupervised training tasks that are not replicable through supervised training.

For CIFAR-10 MoCo converges much faster and reaches a training accuracy of 87.42% compared to the randomly initialized ResNet-50 when was only able to obtain an accuracy of 73.72%.

While neither model preformed significantly well on the Caltech-101 dataset, MoCo was able to learn better on the small training set then ResNet-50. These models most likely did not perform well due to the limited size of the dataset.

Training on the STL-10 dataset produced similar results to training on the Caltech dataset. There is a clear difference between the two models with MoCo performing better both accuracy and convergence rate wise. The limited size of the dataset is also the most likely culprit as to why the accuracies are so low.

Using transfer learning increases the convergence rate and accuracy. This method does seem to require large amounts of compute resources since MoCo took ~6 days to pre-train on 64 GPUs, however when trained it has relatively fast fine-tuning time.



# SimCLR
We fine-tuned a pre-trained SimCLR (v1) model on the three image classifiation datasets. The results can be replicated by running thefollowing commands:
```
git checkout kball/simclr
python convert.py [path-to-simclr-ckpt]
python eval.py -d DATASET
```

We downloaded the pre-trained SimCLR 1x model checkpoints from [the SimCLR github](https://github.com/google-research/simclr) and converted them to pytorch using the following [SimCLR pytorch converter](https://github.com/tonylins/simclr-converter). Finally we add a randomly initialized linear layer with output dimension equal to the class size for each dataset.

We use the training and test splits provided by the `torchvision.datasets` library for each dataset.

All models were trained using cross-entropy loss and Adam optimizer with a learning rate of 1e-3.

## CIFAR-10

![train-cifar](images/simclr/CIFAR-10_default_acc_train.png) ![test-cifar](images/simclr/CIFAR-10_default_acc_val.png)

## Caltech-101

![train-cal](images/simclr/Caltech-101_default_acc_train.png) ![test-cal](images/simclr/Caltech-101_default_acc_val.png)

## STL-10

![train-stl](images/simclr/STL-10_default_acc_train.png) ![test-stl](images/simclr/STL-10_default_acc_val.png)

## Observations

Overall the (smallest) pretrained SimCLR model seems to work really well when finetuned on image classification datasets. My result on the Caltech-101 dataset exceeds the paper's reported result (which uses a larger pretrained ResNet), so I feel like it must be spurious, but I'm not sure what differed in my experiment. The CIFAR-10 seems congruous with the paper's results; they did not evaluate on STL-10 so I have no comparison for that result.

# VirTex
The VirTex paper uses a natural language caption associated with a particular image to help with representational learning behind images. They take captions and use the embeddings generated to do a pre-text task. They show that the weights learned by the resnet model can be transfered to downstream tasks leading to great performance increase. While in the paper they test many downstream tasks, we use the representations for doing the downstream task of image classification.


## CIFAR-10
![acc-cifar-20epochs](images/VirTexCIFAR.png)

## Caltech-101

![train-cal](images/VirTexCaltech101.png)

## STL-10
![acc-cifar-20epochs](images/VirTexSTL10.png)

## Observations

| Datasets/Model | VirTex | ResNet-50 |
|---|---|---|
| CIFAR-10 | 91.4 | 73.72 |
| Caltech-101 | 24.8 | 18.05 |
| STL10 | 57.4 | 23.28 |

The pretrained model does better on the downstream task of image classification than the baseline ResNet50 model. The model converges faster than the earlier models. The test accuracy as well as the train accuracy has a large outperformance than the baseline. It is important to not that the model does better on particular domains. For example, it does much better in the domain of CIFAR-10 than the baseline as well as the other models. This might be because the pretraining task could allow the model to represent complex environments better.


# ConVIRT
The paper Contrastive Learning of Medical Visual Representations from Paired Images and Text has no official published code. A Pytorch implementation for ConVIRT is available at [here](https://github.com/edureisMD/ConVIRT-pytorch).

The authors use the [MIMIC-CXR database](https://physionet.org/content/mimic-cxr/2.0.0/) for contrastive learning. This is a large database including X-Ray images from multiple patients, along text descriptions. Unfortunately the database requires approval which can take up to 4 weeks.

A similar database was found, [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/), which was used to illustrate the pruposes of the original paper. The database consists of 224,316 chest radiographs of 65,240 patients.


## CIFAR-10
![acc-cifar-20epochs](images/ConVIRT-SimCLR-CIFAR10-20epochs.PNG) ![loss-cifar](images/ConVIRT-SimCLR-CIFAR10-20epochs-loss.PNG)
## STL-10
![acc-stl-20epochs](images/ConVIRT-SimCLR-STL10-20epochs.PNG) ![loss-stl](images/ConVIRT-SimCLR-STL10-20epochs-loss.PNG)

### 100 epochs run
<img src="images/ConVIRT-SimCLR-CIFAR10.PNG" width="500" height="500"> <img src="images/ConVIRT-SimCLR-STL10.PNG" width="500" height="500">

## CheXpert
Sample Data:

![chexpert-data](images/cheXpert-sample-data.PNG)

Training on 2 epochs:

![chexpert-2epochs](images/cheXpert-results-2epochs.PNG)

The model was only trained for 2 epochs due to time constraints. Will attempt to train for longer and update plot by Monday night.


# Comparison

### Accuracy comparsion of contrastive learning models between datasets (trained till 20 epochs)

| Datasets/Model | MoCo | SimCLR | VirTex | ConVIRT |
|---|---|---|---|---|
| CIFAR-10 | 87.42 | 82.26 | 91.4 | 63.23  |
| Caltech-101 | 25.08 | 99.02  | 24.8 | - |
| STL10 | 54.45 | 85.81 | 57.4 | 66.145 |

# References

[MoCo pretrained weights](https://github.com/facebookresearch/moco)

[MoCo weights loading example](https://discuss.pytorch.org/t/how-to-load-moco-model-weights-that-are-stored-as-an-state-dict/111549/4)

[SimCLR pretrained weights](https://github.com/google-research/simclr)

[SimCLR pytorch converter](https://github.com/tonylins/simclr-converter)

[ConVIRT repository](https://github.com/edureisMD/ConVIRT-pytorch)
