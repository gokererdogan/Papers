### Design choices
- Loss functions should not care about batch size. They should always return a vector of length batch_size.
- Loss functions should not average over non-batch dimensions. Unfortunately, this is the default in mxnet.
- Trainer takes care of dividing by the batch size.
- Report number of samples during training, not number of batch updates, or number of epochs. So you can plot results with different batch sizes.
- Similarly, take in the number of training samples as parameter (not number of updates or epochs).
  - Epoch might not make sense for datasets with infinite amount of data.
- Having batch size in forward methods is not a good idea because then you'd need to change the batch size in the model (and in all the members that use batch size)if you want to run the same model with a different batch size. FIX THIS in the current source code at some point.

### To think about
How much unittests?
- For now, only the most critical pieces like the loss functions.
- No integration tests yet.

Who should be responsible for plotting losses etc.?
- We can let the `train` function do that but for different plots, different code is needed. would look ugly.
- We can let the `train_step` `val_step` functions do these. these functions implement a single step of training/validation.
  - These functions would take in plotter (`sw`) and plot whatever they want
  - Ultimately, we can have a trainer/algorithm class that implements train and validation steps, and that class can take care of plotting
- Or we can have a callback design where train function takes callbacks that do the plotting and pass in the model, plotter, and iteration number to these callbacks.

Add random projections to check gradient like in theano to support functions with non-scalar outputs?

Should I constrain gx_tilde and gy_tilde? pass through tanh for example. in the paper, they are not constrained but this will produce means outside of image bounds. (these are not constrained in other implementations online)


I don't want this library to be overdesigned; in fact, I don't want it to be a library, more of a collection of different models. The rationale for this is that I often find myself spending a lot of time on design. But good software design is not the aim of this project. It is to just implement interesting models (often as quickly as possible) But at the same time, I can't resist the urge to design somewhat, for getting rid of some obvious duplication. For example. the train function and the plotting callbacks are one example where a bit of design helps. But apart from that, I am not going to design the classes in separate papers into a unified whole. This can be done but it is not my aim in this project. (So occasionally you will see some ugly code and unnecessary duplication.)
