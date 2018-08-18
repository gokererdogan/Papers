In progress...

An [MXNet](https://mxnet.incubator.apache.org/) implementation of the recurrent latent variational autoencoder model DRAW proposed in [1]. 

`core.py` implements the model and loss function. 

`train_mnist.py`trains DRAW on MNIST dataset. Selective attention is **not** implemented yet. 

```
usage: train_mnist.py [-h] [--batch_size BATCH_SIZE]
                      [--input_height INPUT_HEIGHT]
                      [--input_width INPUT_WIDTH] [--num_steps NUM_STEPS]
                      [--num_recurrent_units NUM_RECURRENT_UNITS]
                      [--latent_dim LATENT_DIM]
                      [--learning_rate LEARNING_RATE]
                      [--num_train_samples NUM_TRAIN_SAMPLES]
                      [--num_val_samples NUM_VAL_SAMPLES]
                      [--val_freq VAL_FREQ] [--gpu]

Train DRAW on MNIST dataset

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size
  --input_height INPUT_HEIGHT, -hh INPUT_HEIGHT
                        Height of input image
  --input_width INPUT_WIDTH, -ww INPUT_WIDTH
                        Height of input image
  --num_steps NUM_STEPS, -s NUM_STEPS
                        Height of input image
  --num_recurrent_units NUM_RECURRENT_UNITS, -u NUM_RECURRENT_UNITS
                        Number of units in recurrent encoder and decoder
  --latent_dim LATENT_DIM, -l LATENT_DIM
                        Latent space dimension (number of elements)
  --learning_rate LEARNING_RATE, -r LEARNING_RATE
                        Learning rate
  --num_train_samples NUM_TRAIN_SAMPLES, -t NUM_TRAIN_SAMPLES
                        Number of training samples
  --num_val_samples NUM_VAL_SAMPLES, -v NUM_VAL_SAMPLES
                        Number of validation samples (per validation run)
  --val_freq VAL_FREQ, -f VAL_FREQ
                        Validation frequency (run validation every val_freq
                        training samples)
  --gpu                 If True, train on GPU
```

This script uses `tensorboard` to plot training/validation metrics. Point `tensorboard` to `results` under this folder.
```
tensorboard --logdir=results
```


[1] K. Gregor, I. Danihelka, A. Graves, D. J. Rezende, and D. Wierstra, “DRAW: A Recurrent Neural Network For Image Generation,” arXiv:1502.04623 [cs], Feb. 2015.

