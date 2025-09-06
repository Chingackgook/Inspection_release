# API Documentation for SARIMAModel

## SARIMAModel

### Description
The `SARIMAModel` class implements a Seasonal Autoregressive Integrated Moving Average (SARIMA) model for time series forecasting. It provides methods for fitting the model, making predictions, and retrieving the default parameter search space.

### Attributes
- **data**: `TimeSeriesData`
  - The input time series data for the model.
  
- **params**: `SARIMAParams`
  - The parameters for the SARIMA model.

- **start_params**: `Optional[npt.NDArray]`
  - Initial guess for the solution for the loglikelihood maximization.

- **transformed**: `Optional[bool]`
  - Indicates whether the start parameters are already transformed.

- **includes_fixed**: `Optional[bool]`
  - Indicates whether the start parameters include fixed parameters.

- **cov_type**: `Optional[str]`
  - Method for calculating the covariance matrix of parameter estimates.

- **cov_kwds**: `Optional[Dict[str, Any]]`
  - Arguments for covariance matrix computation.

- **method**: `Optional[str]`
  - Solver method from `scipy.optimize` to be used.

- **maxiter**: `Optional[int]`
  - Maximum number of iterations to perform.

- **full_output**: `Optional[bool]`
  - Indicates whether to have all available output in the Results object’s `mle_retvals` attribute.

- **disp**: `Optional[bool]`
  - Indicates whether to print convergence messages.

- **callback**: `Optional[Callable[[npt.NDArray], None]]`
  - Callable object to be called after each iteration.

- **return_params**: `Optional[bool]`
  - Indicates whether to return only the array of maximizing parameters.

- **optim_score**: `Optional[str]`
  - Method for calculating the score vector.

- **optim_complex_step**: `Optional[bool]`
  - Indicates whether to use complex step differentiation.

- **optim_hessian**: `Optional[str]`
  - Method for numerically approximating the Hessian.

- **low_memory**: `Optional[bool]`
  - Indicates whether to reduce memory usage.

- **model**: `Optional[MLEResults]`
  - The fitted SARIMA model.

- **include_history**: `bool`
  - Indicates whether to include historical data in predictions.

- **alpha**: `float`
  - Confidence level for prediction intervals.

- **fcst_df**: `Optional[pd.DataFrame]`
  - DataFrame containing forecast results.

- **freq**: `Optional[float]`
  - Frequency of the time series.

- **y_fcst**: `Optional[npt.NDArray]`
  - Forecasted values.

- **y_fcst_lower**: `Optional[npt.NDArray]`
  - Lower bounds of the forecasted values.

- **y_fcst_upper**: `Optional[npt.NDArray]`
  - Upper bounds of the forecasted values.

- **dates**: `Optional[pd.DatetimeIndex]`
  - Dates corresponding to the forecasted values.

### __init__

#### Description
Initializes the SARIMAModel with the provided time series data and parameters.

#### Parameters
- **data**: `TimeSeriesData`
  - The input time series data. Must be univariate.
  
- **params**: `SARIMAParams`
  - The parameters for the SARIMA model.

#### Returns
- `None`

### fit

#### Description
Fits the SARIMA model using maximum likelihood estimation via the Kalman filter.

#### Parameters
- **start_params**: `Optional[npt.NDArray]`
  - Initial guess for the solution for the loglikelihood maximization.
  
- **transformed**: `bool`
  - Indicates whether the start parameters are already transformed. Default is `True`.

- **includes_fixed**: `bool`
  - Indicates whether the start parameters include fixed parameters. Default is `False`.

- **cov_type**: `Optional[str]`
  - Method for calculating the covariance matrix. Default is `'opg'`.

- **cov_kwds**: `Optional[Dict[str, Any]]`
  - Arguments for covariance matrix computation.

- **method**: `str`
  - Solver method from `scipy.optimize`. Default is `'lbfgs'`.

- **maxiter**: `int`
  - Maximum number of iterations. Default is `50`.

- **full_output**: `bool`
  - Indicates whether to have all available output. Default is `True`.

- **disp**: `bool`
  - Indicates whether to print convergence messages. Default is `False`.

- **callback**: `Optional[Callable[[npt.NDArray], None]]`
  - Callable object to be called after each iteration.

- **return_params**: `bool`
  - Indicates whether to return only the array of maximizing parameters. Default is `False`.

- **optim_score**: `Optional[str]`
  - Method for calculating the score vector.

- **optim_complex_step**: `bool`
  - Indicates whether to use complex step differentiation. Default is `True`.

- **optim_hessian**: `Optional[str]`
  - Method for numerically approximating the Hessian.

- **low_memory**: `bool`
  - Indicates whether to reduce memory usage. Default is `False`.

#### Returns
- `None`

### predict

#### Description
Generates forecasts from the fitted SARIMA model.

#### Parameters
- **steps**: `int`
  - Number of forecast steps to generate.

- **exog**: `Optional[ArrayLike]`
  - Exogenous variables to be used in the forecast.

- **include_history**: `bool`
  - Indicates whether to include historical data in the output. Default is `False`.

- **alpha**: `float`
  - Confidence level for prediction intervals. Default is `0.05`.

- **kwargs**: `Any`
  - Additional keyword arguments.

#### Returns
- `pd.DataFrame`
  - A DataFrame containing the forecasted values and their confidence intervals.

### get_parameter_search_space

#### Description
Retrieves the default parameter search space for the SARIMA model.

#### Parameters
- `None`

#### Returns
- `List[Dict[str, Any]]`
  - A list of dictionaries representing the default SARIMA parameter search space.

