# Character-level Language Modelling with LSTMs in PyTorch
The blog post around this project can be found [here](https://medium.com/@dhananjaytomar/using-benchmark-datasets-character-level-language-modeling-ef16afa21101)

This repository contains some (hopefully clean) code for loading benchmark datasets as PyTorch's Dataset objects and 
training LSTMs on those datasets. You may find the code in [dataloader.py](/blob/master/dataloader.py) to be most 
useful for your research.

## Training the baseline model
You can train the default model defined in [model.py](/blob/master/model.py) for a particular dataset by running

`python train.py --dataset <dataset_name>`

The downloaded datasets will stored in the `data` directory.

## Adding more datasets
### Datasets similar to text8
If you wish to datasets similar to the ones already present i.e. datasets where the whole train/valid/test data can be 
treated as a single sequence, then you can simply extend `LanguageModelingDataset` class. In particular, you'll only have 
to think about:

**1. Dataset's vocabulary**: What all symbols are there in the training data? Can they be hard-coded like it has 
been done in for the text8 and Penn Treebank dataset classes or do they need to be initialised from the training data 
like it has been done for the Hutter Prize dataset?

**2. Downloading dataset**: From where can be dataset downloaded? Does it need to be decompressed?

**3. Loading dataset**: How should the data be split into train, valid and test sets if it comes without those splits?

### Word-level datasets
Adding word-level datasets should only require you to pass the data in `_prepare_data` function as a list of words 
instead of a list/string of characters. Note that the assumption still is that the whole dataset can be treated as a 
single sequence.

### Datasets with multiple training sequences
If the dataset contains many different sequences and you do not wish to collapse them into a single sequence, you'll 
have to modify the `_prepare_data` function. Depending on the dataset, you might just have to delete some lines or it 
might require more effort but the effort should still be low.