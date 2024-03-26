# Temporal_CNN

A work in progress...

## Convolutional Neural Networks

Convolutional Neural Networks (CNN's) are commonly used in image classifcation tasks because they are able to take into account the spatial structure of an image.  This contrasts with a standard neural network, which would treat every pixel as an independent feature.

Just as a 2-D image has red, green and blue channels, which are best interpreted when analysed together, a price-data time-series can be assigned channels representing open/high/low/close prices, and traded volume.

By treating a price-data time-series as a 1-dimensional image, we hope to improve the model's ability to recognise temporal correlations, and therefore predict profitable trading opportunities.  

Whilst attempting to predict market movements is a hard problem due to their highly stochastic nature, even weak models can provide enough of a 'gambler's edge' to be profitable, assuming a large enough number of trades, and providing certain caveats are accounted for.

## Multi-modal Networks

It may be true that markets behave differently at different times of the day or week (or year, in some cases).  By expanding the index of the time series into a one-hot encoded categorical dataset, we can feed this into a separate branch of the CNN, in the hope of increasing its predictive power.

## 1_data_preparation.py

Converts 'open-high-low-close' (OHLC) price data into a format suitable for machine learning.  It uses first-order filters and significant levels to extract learnable features from the raw data.

Target labels are generated according to (potential) profitable trading conditions being met, whilst time and day labels are separated out and one-hot encoded.  

The working dataframe is then transformed into a tensor of dimensions [batch_size, steps, channels] by using a rolling window method.  Appropriate category labels are selected to match the last step in each window, and transformed into a tensors of dimensions [batch_size, categories].

## To Do list
Optimise model
Generate precision-recall curves for target classes
Generate confusion matrix for chosen thresholds
Calculate profit factor and drawdown