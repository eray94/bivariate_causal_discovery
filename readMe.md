# Causal Discovery on Bivariate Data

Predict the cause-effect relationships on these bivariate data sets.

As there are two co-variates (A,B), this is a binary causal discover
 problem; the outcome is either 0: A→B or 1: B→A.

### Data

Database with cause-effect pairs(Tuebingen) can be 
found [here](http://webdav.tuebingen.mpg.de/cause-effect/).

Raw data and data description files are can be found from above link. 
Description files explain the data and ground truth of the causal 
relations.


In this project, causal relation of first 21 dataset were predicted.

```In all 21 dataset, causal relation from A→B. In order to see AUC score in 10 dataset A and B values are swap.```

 
Raw data files which have odd number have A→B relation and files which
have event number have B→A. (i.e. pair0001.txt has A→B relation,
pair0002.txt has B→A relation and so on.)

Ground truths are created accordingly.

## Implementation

### Algorithm

In this project, cause-effect relationships on these bivariate data sets 
are predicted by implementing Algorithm 1 of Mooij et al.

Paper can be found here:
[Distinguishing Cause from Effect Using Observational Data:
Methods and Benchmarks"](https://jmlr.org/papers/v17/14-518.html)

![Algorithm](./readme_images/mooij_algo_1.png?raw=true "Algorithm")

### Implementation Details

As the base regression algorithm (steps 1(a) and 1(b)),
 simple neural networks with same architecture were used.

For statistical independence testing (steps 3(a) and 3(b)),
 use the Hilbert-Schmidt Information Criterion(HSIC) was used.
 
Ground truths are obtained from description files of the data.

#### Evaluation Metrics
1) Accuracy
2) Area Under Receiver Operating Characteristics Curve (AUC)



## Installation 

Create Virtual Environment(Optional but Recommended)

    $ pip3 install virtualenv
    $ virtualenv [virtualenv_name]
    $ source [virtualenv_name]/bin/activate


Install Requirements

    $ pip3 install -r requirements.txt

## Run

### Causal Discovery

Neural networks' weights are stored in model_weights folder.
If folder is empty or corrupt please train them again.

In order to test causal discovery with pre-trained networks,
 simply run following command: 

    $ python discover.py

For test part, neural networks only used for prediction and residuals 
calculated. Then HSIC scores are obtained and causal relations are predicted.

Evaluation metrics printed.

### Train Regression Models


Train 21 x 2 simple neural networks for regression which explained in 
algorithm section.

If you want to retrain networks, follow these steps:

In this part, using a ```fixed seed``` is crucial.

1) Seed is determined.
2) Test(10%) and train(90%) data split. Ratios can be changed.
3) For each dataset, 2 neural networks are trained.
 One for A to B and the other one for B to A. In total 42.
4) Model weights are stored. Simply run the train.py, new weight will 
overwrite on the existing files.
5) In order the extend dataset, you can simply add new data files, but 
```ground truth must also be updated```.

NOTE: ```If you run it, you will loose the current model weights. ```

In order to train regression models, simply run this command: 

    $ python train.py


Train all model takes approximately 1 minute on CPU.

All models will train and weights' of models will store.

Evaluation metrics are also printed.


## Results

In all 21 datasets, variable A is the cause of variable B. 
While the datasets with an odd number of filenames preserved
this situation, the directions were reversed for even ones.
This case is to calculate the AUC score.
The number of datasets which is 20 was increased to 21 in order that
the labels in the ground truth were not equal to each other.
The differences between such AUC and Accuracy values ​​
became slightly more visible.

Results vary for datasets. Finding causal relations in some dataset
is quite easy compared to others. The first 21 dataset were used
instead of selecting easily identifiable datasets in terms of providing
an objective result. The dataset number can be easily increased or decreased.
Just add the related files and edit the ground truth constant.

In addition, the results may vary with each train. 
For some datasets, independence score is very close to each other
and different estimates can be made as a result of different trainings. 
A confidence level can be set for these results, but this is beyond 
the scope of the assignment.

The results of the 15 different trainings are as follows:

![Box plot](./readme_images/ci_res_box.png?raw=true "Box plots of results")
![Stats](./readme_images/ci_res_num.png?raw=true "Statistics of results")


Current weights are belong the one of the most successful predictions.
Accuracy and AUC slightly higher than 80%.

You can see these results just running discover.py and also you can train
your models from start just running train.py. New weights will be overwrite
on the previous ones. Most probably you will obtain Accuracy and AUC results 
between 61% and 81%.


---
```Istanbul Technical University```

```Computer Engineering M.Sc. - Causal Inference(BLG553E)```

```Fall/2020```

---

```Eray Mert Kavuk```





