from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#################################################
################### LOAD DATA ###################

file = 'AirPassengers.csv'

df = pd.read_csv(file)

###########################################
################### EDA ###################

# Basic data info
print(df.info())
print(df.describe())
print(df.head())

# Rename columns for clarity
df.rename(columns={'#Passengers': 'AirPassengers'}, inplace=True)
print(df.head())

# Convert date object to datetime type
df['Month'] = pd.DatetimeIndex(df['Month'])
print(df.dtypes)

# Rename columns to conform to Prophet input columns. ds (the time column) and y (the metric column)
df = df.rename(columns={'Month': 'ds',
                        'AirPassengers': 'y'})
print(df.head())

# Visualize the data
ax = df.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Monthly Number of Airline Passengers')
ax.set_xlabel('Date')

plt.show()

########################################################
################### CREATE THE MODEL ###################

# set the uncertainty interval to 95% (the Prophet default is 80%)
model = Prophet(interval_width=0.95)

# Fit the model
model.fit(df)

# Create future dataframe
# We instructed Prophet to generate 36 datestamps in the future. This can be changed as needed.
# Because we are working with monthly data, we clearly specified the desired frequency of the timestamps (in this case, MS is the start of the month)
future_dates = model.make_future_dataframe(periods=36, freq='MS')
future_dates.head()

# Predict future values
forecast = model.predict(future_dates)

# Prophet returns a large DataFrame with many interesting columns, but we subset our output to the columns most relevant to forecasting. These are:
# ds: the datestamp of the forecasted value
# yhat: the forecasted value of our metric (in Statistics, yhat is a notation traditionally used to represent the predicted values of a value y)
# yhat_lower: the lower bound of our forecasts
# yhat_upper: the upper bound of our forecasts
# A variation in values from the output presented is to be expected as Prophet relies on Markov chain Monte Carlo (MCMC) methods to generate its forecasts.
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

# Plot the results of our forecasts
# Prophet plots the observed values of our time series (the black dots), the forecasted values (blue line) and the uncertainty intervals of our forecasts (the blue shaded regions).
model.plot(forecast, uncertainty=True)
plt.show()

##########################################################################
################### PLOTTING THE FORECASTED COMPONENTS ###################
# Plot components of the forecast
# The first plot shows that the monthly volume of airline passengers has been linearly increasing over time.
# The second plot highlights the fact that the weekly count of passengers peaks towards the end of the week and on Saturday.
# The third plot shows that the most traffic occurs during the holiday months of July and August.
model.plot_components(forecast)
plt.show()

######################################################################
################### Adding ChangePoints to Prophet ###################
# Changepoints are the datetime points where the time series have abrupt changes in the trajectory.
# By default, Prophet adds 25 changepoints to the initial 80% of the data-set.
# Let’s plot the vertical lines where the potential changepoints occurred.
fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast)
plt.show()
# Output the dates where changepoints occurred
print(model.changepoints)

# We can change the inferred changepoint range by setting the changepoint_range
# The number of changepoints can be set by using the n_changepoints parameter when initializing prophet.
pro_change= Prophet(n_changepoints=20, yearly_seasonality=True)
forecast = pro_change.fit(df).predict(future_dates)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)
plt.show()

#######################################################
################### ADJUSTING TREND ###################
# Prophet allows us to adjust the trend in case there is an overfit or underfit.
# changepoint_prior_scale helps adjust the strength of the trend.
# Default value for changepoint_prior_scale is 0.05.
# Decrease the value to make the trend less flexible.
# Increase the value of changepoint_prior_scale to make the trend more flexible.
# Increasing the changepoint_prior_scale to 0.08 to make the trend flexible.
pro_change= Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.08)
forecast = pro_change.fit(df).predict(future_dates)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)
plt.show()

# Decreasing the changepoint_prior_scale to 0.001 to make the trend less flexible.
pro_change= Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.001)
forecast = pro_change.fit(df).predict(future_dates)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)
plt.show()
