# kTsnn
Implementation in keras of some neural networks related with time series short and long-term forecasting. 

The main idea is to fold a time-series dataset to have in the same row multiple "lags" of each column. Then, we use the lagged columns to predict the future ones. For long-term forecasting, we use the predictions as evidence for the next step.

This aims to be a performance comparison with my Gaussian dynamic Bayesian network (https://github.com/dkesada/dbnR/) model. I want to compare my GDBN model with a NN model in similar ground and in the process create a kind of "plug-and-play" alternative in case I need it in the future.

# References
- Time delay neural networks: https://en.wikipedia.org/wiki/Time_delay_neural_network
- Keras: https://keras.io/
