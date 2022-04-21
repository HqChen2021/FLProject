# FLProject
This repository contains code for the FL project. There're several .py files included.
## FL_main.py 
The main function of the model. Sampling clients to participant in current round, aggregating weights uploaded by clients.
## utils.py 
Download dataset (like MNIST, FMNIST, and CIFAR)
## sampling.py 
Sampling client data from dataset (like MNIST, FMNIST, and CIFAR). Two stratgies are available, iid and non-iid (for the non-iid sampling, the data is sampled according dirichlet distribution) 
For example.
![2071650420242_ pic](https://user-images.githubusercontent.com/87748244/164381804-16f27f7e-c907-47f5-91a1-e80367ec3ace.jpg)
## models.py
contains three different models
## options.py
experiment configuration parameters
## updata.py
local updating
