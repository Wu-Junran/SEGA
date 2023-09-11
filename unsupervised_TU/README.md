## Dependencies

* Python 3.7
* PyTorch 1.8
* PyTorch Geometric 2.0.1

Then, you need to create directories for data and recoreding finetuned results to avoid errors:

```
mkdir data
mkdir logs
```

## Training & Evaluation

```
python treecl.py -d $DATASET_NAME --aug $AUGMENTATION ...
```

* ```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/).
* ```$AUGMENTATION``` is the augmentation method proposed by GraphCL for unsuervised learning.
* ```...``` are the other hyper-parameters mentioned in the paper.


Produce the results in paper:
```
python show_acc.py
```

## Acknowledgements

The backbone implementation is reference to https://github.com/Shen-Lab/GraphCL/tree/master/unsupervised_TU.
