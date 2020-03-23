# weather_encoders

Experiments with CNN encoder-decoder for weather data.

Testing differentiable categorical loss functions for modeling rainfall.

Main model is `unet.py` and differentiable loss functions are defined inside `categorical_losses`.

ERA5 data can be used to reproduce the experiments available [here](https://climate.copernicus.eu/climate-reanalysis)
