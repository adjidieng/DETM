# DETM

This is code that accompanies the paper titled "The Dynamic Embedded Topic Model" by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. (Arxiv link: https://arxiv.org/abs/1907.05545).

The DETM is an extension of the Embedded Topic Model (https://arxiv.org/abs/1907.04907) to corpora with temporal dependencies. The DETM models each word with a categorical distribution whose parameter is given by the inner product between the word embedding and an embedding representation of its assigned topic at a particular time step. The word embeddings allow the DETM to generalize to rare words. The DETM learns smooth topic trajectories by defining a random walk prior over the embeddings of the topics. The DETM is fit using structured amortized variational inference with LSTMs.

## Dependencies

+ python 3.6.7
+ pytorch 1.1.0

## Datasets

The pre-processed UN and ACL datasets can be found below:

+ https://bitbucket.org/franrruiz/data_acl_largev/src/master/
+ https://bitbucket.org/franrruiz/data_undebates_largev/src/master/

The pre-fitted embeddings can be found below:

+ https://bitbucket.org/diengadji/embeddings/src

All the scripts to pre-process a dataset can be found in the folder 'scripts'. 

## Example

To run the DETM on the ACL dataset you can run the command below. You can specify different values for other arguments, peek at the arguments list in main.py.

```
python main.py --dataset acl --data_path PATH_TO_DATA --emb_path PATH_TO_EMBEDDINGS --min_df 10 --num_topics 50 --lr 0.0001 --epochs 1000 --mode train
```


## Citation
```
@article{dieng2019dynamic,
  title={The Dynamic Embedded Topic Model},
  author={Dieng, Adji B and Ruiz, Francisco JR and Blei, David M},
  journal={arXiv preprint arXiv:1907.05545},
  year={2019}
}
```


