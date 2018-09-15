**Without attention**

 ![draw_wo_attn_0](../assets/draw_wo_attn_0.gif)  ![draw_wo_attn_1](../assets/draw_wo_attn_1.gif)  ![draw_wo_attn_2](../assets/draw_wo_attn_2.gif)  ![draw_wo_attn_3](../assets/draw_wo_attn_3.gif)  ![draw_wo_attn_4](../assets/draw_wo_attn_4.gif) ![draw_wo_attn_5](../assets/draw_wo_attn_5.gif)  ![draw_wo_attn_6](../assets/draw_wo_attn_6.gif)  ![draw_wo_attn_7](../assets/draw_wo_attn_7.gif)  ![draw_wo_attn_8](../assets/draw_wo_attn_8.gif)  ![draw_wo_attn_9](../assets/draw_wo_attn_9.gif) 

**With attention**

 ![draw_w_attn_0](../assets/draw_w_attn_0.gif)  ![draw_w_attn_1](../assets/draw_w_attn_1.gif)  ![draw_w_attn_2](../assets/draw_w_attn_2.gif) ![draw_w_attn_3](../assets/draw_w_attn_3.gif)  ![draw_w_attn_4](../assets/draw_w_attn_4.gif) ![draw_w_attn_5](../assets/draw_w_attn_5.gif)  ![draw_w_attn_6](../assets/draw_w_attn_6.gif)  ![draw_w_attn_7](../assets/draw_w_attn_7.gif) ![draw_w_attn_8](../assets/draw_w_attn_8.gif)  ![draw_w_attn_9](../assets/draw_w_attn_9.gif) 



An [MXNet](https://mxnet.incubator.apache.org/) implementation of the recurrent latent variational autoencoder model DRAW proposed in [1]. 

`core.py` implements the model (with and without selective attention) and loss function. This implementation differs from the model presented in the paper in one minor respect: I don't learn the initial canvas. I found that this actually produces less visually appealing results.

`train_mnist.py`trains DRAW on MNIST dataset.  

In the figure above, you see samples from two models, one with attention and another without attention, trained on MNIST with the default parameters (for 500,000 training samples instead 1,000,000). Note these are true samples from the model, not the reconstructions for real images in the training or validation sets.

```
usage: train_mnist.py [-h] [--batch_size BATCH_SIZE]
                      [--input_height INPUT_HEIGHT]
                      [--input_width INPUT_WIDTH] [--num_steps NUM_STEPS]
                      [--num_recurrent_units NUM_RECURRENT_UNITS]
                      [--latent_dim LATENT_DIM]
                      [--learning_rate LEARNING_RATE]
                      [--num_train_samples NUM_TRAIN_SAMPLES]
                      [--num_val_samples NUM_VAL_SAMPLES]
                      [--val_freq VAL_FREQ] [--gpu] [--attention]
                      [--read_size READ_SIZE] [--write_size WRITE_SIZE]
                      [--logdir LOGDIR]

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
  --attention           If True, train with selective attention.
  --read_size READ_SIZE, -ar READ_SIZE
                        If True, train with selective attention.
  --write_size WRITE_SIZE, -aw WRITE_SIZE
                        If True, train with selective attention.
  --logdir LOGDIR       Log directory for mxboard.
```

This script uses `tensorboard` to plot training/validation metrics. Point `tensorboard` to log directory you specified.
```
tensorboard --logdir=results
```

[1] K. Gregor, I. Danihelka, A. Graves, D. J. Rezende, and D. Wierstra, “DRAW: A Recurrent Neural Network For Image Generation,” arXiv:1502.04623 [cs], Feb. 2015.

