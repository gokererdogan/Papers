
 ![convdraw_0](../assets/convdraw_0.png)  ![convdraw_1](../assets/convdraw_1.png)  ![convdraw_2](../assets/convdraw_2.png)  ![convdraw_3](../assets/convdraw_3.png)  ![convdraw_4](../assets/convdraw_4.png) ![convdraw_5](../assets/convdraw_5.png)  ![convdraw_6](../assets/convdraw_6.png)  ![convdraw_7](../assets/convdraw_7.png)  ![convdraw_8](../assets/convdraw_8.png)  ![convdraw_9](../assets/convdraw_9.png) 

 ![convdraw_10](../assets/convdraw_10.png)  ![convdraw_11](../assets/convdraw_11.png)  ![convdraw_12](../assets/convdraw_12.png) ![convdraw_13](../assets/convdraw_13.png)  ![convdraw_14](../assets/convdraw_14.png) ![convdraw_15](../assets/convdraw_15.png)  ![convdraw_16](../assets/convdraw_16.png)  ![convdraw_17](../assets/convdraw_17.png) ![convdraw_18](../assets/convdraw_18.png)  ![convdraw_19](../assets/convdraw_19.png) 

![convdraw_20](../assets/convdraw_20.png)  ![convdraw_21](../assets/convdraw_21.png)  ![convdraw_22](../assets/convdraw_22.png)  ![convdraw_23](../assets/convdraw_23.png)  ![convdraw_24](../assets/convdraw_24.png)  ![convdraw_25](../assets/convdraw_25.png)  ![convdraw_26](../assets/convdraw_26.png)  ![convdraw_27](../assets/convdraw_27.png)  ![convdraw_28](../assets/convdraw_28.png)  ![convdraw_29](../assets/convdraw_29.png)  

![convdraw_gif_0](../assets/convdraw_0.gif)  ![convdraw_gif_1](../assets/convdraw_1.gif)  ![convdraw_gif_2](../assets/convdraw_2.gif)  ![convdraw_gif_3](../assets/convdraw_3.gif)  ![convdraw_gif_4](../assets/convdraw_4.gif)  ![convdraw_gif_5](../assets/convdraw_5.gif)  ![convdraw_gif_6](../assets/convdraw_6.gif)  ![convdraw_gif_7](../assets/convdraw_7.gif)  ![convdraw_gif_8](../assets/convdraw_8.gif)  ![convdraw_gif_9](../assets/convdraw_9.gif)  




An [MXNet](https://mxnet.incubator.apache.org/) implementation of the recurrent latent variational autoencoder model ConvDRAW proposed in [1].

`core.py` implements the model and loss function. 

`train_omniglot.py`trains DRAW on OmniGlot dataset.  

Look at the `../data/omniglot_to_ndarray.py` script to see how to get the dataset.

In the figure above, you see samples from the model trained on OmniGlot with the default parameters (for 500,000 training samples). The final log negative log likelihood is around 70 nats. This is much lower than what is reported in the paper but note that the dataset here is not the same with the dataset used in the paper. So these numbers are probably not comparable. Note the pictures above are true samples from the model, not the reconstructions for real images in the training or validation sets.

```
usage: train_omniglot.py [-h] [--batch_size BATCH_SIZE]
                         [--num_steps NUM_STEPS]
                         [--num_latent_maps NUM_LATENT_MAPS]
                         [--num_recurrent_maps NUM_RECURRENT_MAPS]
                         [--learning_rate LEARNING_RATE]
                         [--num_train_samples NUM_TRAIN_SAMPLES]
                         [--num_val_samples NUM_VAL_SAMPLES]
                         [--val_freq VAL_FREQ] [--gpu] [--logdir LOGDIR]

Train ConvDRAW on Omniglot dataset

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size
  --num_steps NUM_STEPS, -s NUM_STEPS
                        Number of recurrent steps
  --num_latent_maps NUM_LATENT_MAPS, -l NUM_LATENT_MAPS
                        Number of latent feature maps.
  --num_recurrent_maps NUM_RECURRENT_MAPS, -u NUM_RECURRENT_MAPS
                        Number of feature maps in recurrent encoder and
                        decoder
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
  --logdir LOGDIR       Log directory for mxboard.
```

This script uses `tensorboard` to plot training/validation metrics. Point `tensorboard` to log directory you specified.
```
tensorboard --logdir=results
```

[1] K. Gregor, F. Besse, D. J. Rezende, I. Danihelka, and D. Wierstra, “Towards Conceptual Compression” arXiv:1604.08772 [stat.ML], Apr. 2015.
