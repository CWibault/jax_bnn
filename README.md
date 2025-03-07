# Bayesian Neural Networks in JAX

Implementation of variational inference over the parameters of a feedforward neural network in JAX, closely following the approach of Bayes-by-Backprop [Blundell et al., 2015, [https://arxiv.org/pdf/1505.05424]]. By using a mean-field approximation, the number of parameters in the model is only doubled compared to training a neural network with fixed parameters. The reparameterisation trick is used to sample weights during training.
Amendments to the algorithm in the original paper include the type of prior used (a weak, mean-field Gaussian prior is used rather than a mixture of two Gaussians) and sample frequency (a different \epsilon is sampled for each datapoint in the minibatch, which was found to give smoother gradient updates). 

The codebase is inspired by pyvarinf [https://github.com/ctallec/pyvarinf], which does a similar implementation in PyTorch. 

To test the code, build the Docker Image in the dev file (cd dev, bash build.sh), launch the container (cd .., bash launch_container.sh), and run the test scripts: python3.9 test_vi_1d.py, python3.9 test_vi_2d.py, python3.9 test_vi_posterior_world_model.py. The plots are saved in the runs folder - which can be used to validate results. 

The code was originally written to quantify the uncertainty in the parameters of a World Model for Reinforcement Learning trained using a random-policy-generated offline dataset, which is why the final test script learns a transition World Model for the gymnax CartPole environment [https://github.com/RobertTLange/gymnax]. An extra learn log_sigma term was incorporated into the feedforward neural network to learn a standard deviation over the state transitions, thereby quantifying uncertainty in a potential stochastic environment (CartPole is deterministic). This is unrelated to the standard deviation in network parameters, which quantifies the epistemic (model) uncertainty. The world model training script was included as an example of the posterior variance decreasing with more data, and tracking the KL divergence and posterior variance during training.

NB - if using a Bayesian Neural Network to characterise model uncertainty, using randomised priors rather than Variational Inference was found to give a better representation of the epistemic uncertainty and be more efficient to train (especially for larger networks). The network variance can be characterised using an expectation over the variance in function outputs. 

Please let me know if you have any suggestions - I hope the code can be of use! 
