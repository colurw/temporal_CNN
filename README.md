A work in progress...

## Convolution Neural Networks

Convolutional Neural Networks (CNN's) are typically used in image classifcation tasks because they are able to take into account the spatial structure of an image.  This contrasts with a standard neural network, which would treat every pixel as an independent feature.

Just as a 2-D image has red, green and blue channels, which are best interpreted when analysed together, a price-data time-series can be assigned channels representing open/high/low/close prices, and traded volume.

By treating a price-data time-series as a 1-dimensional image, we hope to improve the model's ability to recognise temporal correlations, and therefore predict profitable trading opportunities.  

Whilst attempting to predict market movements is a hard problem due to their highly stochastic nature, even weak models can provide enough of a 'gambler's edge' to be profitable, assuming a large enough number of trades, and providing certain caveats are accounted for.

## Multi-modal Networks

It may be true that markets behave differently at different times of the day or week (or year, in some cases).  By expanding the index of the time series into a one-hot encoded categorical dataset, we can feed this into a separate branch of the CNN, in the hope of increasing its predictive power.
