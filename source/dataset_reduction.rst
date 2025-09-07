Dataset Reduction for Accelerated Experimentation
=================================================

To streamline the initial development of both centralised and federated
algorithms, only 10% subsets of the training, validation, and test sets
were used for early testing and debugging.

Import required libraries

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

Load data

.. code:: python

    train_images = np.load("/home/mn628/FEDERATED_LEARNING/data/train_images.npy")
    train_masks = np.load("/home/mn628/FEDERATED_LEARNING/data/train_masks.npy")
    val_images = np.load("/home/mn628/FEDERATED_LEARNING/data/val_images.npy")
    val_masks = np.load("/home/mn628/FEDERATED_LEARNING/data/val_masks.npy")
    test_images = np.load("/home/mn628/FEDERATED_LEARNING/data/test_images.npy")
    test_masks = np.load("/home/mn628/FEDERATED_LEARNING/data/test_masks.npy")

Initial shape

.. code:: python

    train_images.shape, train_masks.shape, val_images.shape, val_masks.shape, test_images.shape, test_masks.shape



.. parsed-literal::

    ((20433, 128, 128, 1),
     (20433, 128, 128, 1),
     (1989, 128, 128, 1),
     (1989, 128, 128, 1),
     (7089, 128, 128, 1),
     (7089, 128, 128, 1))


Reduced size

.. code:: python

    new_len_train = len(train_images) // 10
    new_len_val = len(val_images) // 10
    new_len_test = len(test_images) // 10
    
    train_images = train_images[:new_len_train]
    train_masks = train_masks[:new_len_train]
    val_images = val_images[:new_len_val]
    val_masks = val_masks[:new_len_val]
    test_images = test_images[:new_len_test]
    test_masks = test_masks[:new_len_test]

.. code:: python

    train_images.shape, train_masks.shape, val_images.shape, val_masks.shape, test_images.shape, test_masks.shape



.. parsed-literal::

    ((204, 128, 128, 1),
     (204, 128, 128, 1),
     (198, 128, 128, 1),
     (198, 128, 128, 1),
     (708, 128, 128, 1),
     (708, 128, 128, 1))


saving the new dataset

.. code:: python

    np.save("/home/mn628/FEDERATED_LEARNING/data_subset/train_images.npy", train_images)
    np.save("/home/mn628/FEDERATED_LEARNING/data_subset/train_masks.npy", train_masks)
    np.save("/home/mn628/FEDERATED_LEARNING/data_subset/val_images.npy", val_images)
    np.save("/home/mn628/FEDERATED_LEARNING/data_subset/val_masks.npy", val_masks)
    np.save("/home/mn628/FEDERATED_LEARNING/data_subset/test_images.npy", test_images)
    np.save("/home/mn628/FEDERATED_LEARNING/data_subset/test_masks.npy", test_masks)
