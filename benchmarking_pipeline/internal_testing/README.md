# Test Cases & Explanations

Below is a summary of the various test cases I created for the Data Loader, Preprocessor, and Hyperparameter Tuning portions of the pipeline.

## Data Loader Tests Overview

The table below outlines tests I made for the Data Loader portion of the pipeline.

| Test Index               | Its Description                                                                                                                                               |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Test Case 1**          | Load the train split of the synthetically created data_loader_testing dataset chunk 'chunk001.csv', and see if we get the expected list of integers.          |
| **Test Case 2**          | Load the train split of the synthetically created data_loader_testing dataset chunk 'chunk003.csv', and see if we get the expected list of lists of integers. |
| **Implicit Test Case 1** | Run the 'load_several_chunks(...)' command to see if we get the appropriate data_loader_testing dataset chunks, 'chunk001.csv' and 'chunk002.csv'.            |
| **Test Case 3**          | Load the validation split of the synthetically created data_loader_testing dataset chunk 'chunk001.csv', and see if we get the expected list of integers.     |
| **Test Case 4**          | Load the test split of the synthetically created data_loader_testing dataset chunk 'chunk002.csv', and see if we get the expected list of lists of integers.  |


## Preprocessor Tests Overview

The table below outlines tests I made for the Preprocessor portion of the pipeline.

| Test Index      | Its Description                                                                                                                                                                                                                                       |
| :-------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test Case 1** | From the synthetically created preprocessor_testing dataset chunk 'chunk001.csv', ensure that replacing all NaN or None values with the median of the remaining numeric values works and gets us our expected train split.                            |
| **Test Case 2** | From the synthetically created preprocessor_testing dataset chunk 'chunk001.csv', ensure that dropping all rows with NaN or None values, as well as removing outliers with a specified outlier threshold, works and gets us our expected train split. |
| **Test Case 3** | From the synthetically created preprocessor_testing dataset chunk 'chunk002.csv', ensure that normalizing the data using a MinMaxScaler approach, as well as interpolating NaN or None values, works and gets us our expected train split.            |


## Hyperparameter Tuning Tests Overview

The table below outlines tests I made for the Hyperparameter Tuning portion of the pipeline.

| Test Index      | Its Description                                                                                                                                                                                                                                      |
| :-------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test Case 1** | Ensure, using the ARIMA model, that hyperparameter tuning on the test+val splits of the first two chunks of the australian_electricity_demand dataset gets us better performance on the corresponding test splits than using random hyperparameters. |
| **Test Case 2** | Ensure, using the ARIMA model, that hyperparameter tuning on the test+val splits of the first two chunks of the bdg-2_bear dataset gets us better performance on the corresponding test splits than using random hyperparameters.                    |
| **Test Case 3** | Ensure, using the ARIMA model, that hyperparameter tuning on the test+val splits of the first three chunks of the LOOP_SEATTLE dataset gets us better performance on the corresponding test splits than using random hyperparameters.                |

## Evaluator Tests Overview

The table below outlines the tests designed for the Evaluator portion of the pipeline, which is responsible for calculating various performance metrics. These tests verify the correct implementation of the evaluation metrics and the graceful handling of invalid inputs.

| Test Index | Its Description |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Test Case 1** | The `run_tests` function validates the core functionality of the `Evaluator` by computing all of the metrics. It uses predefined true and prediction data to calculate and verify the results for RMSE, MAE, MASE, CRPS, Quantile Loss, and the Interval Score. |
| **Test Case 2** | The `run_test_default_metrics` function ensures that the evaluator can correctly calculate the default metrics, Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), when no specific metrics are requested. |
| **Test Case 3** | The `run_test_missing_required_args` test case confirms that the system raises appropriate `ValueError` exceptions when the necessary arguments for specific metrics are not provided. This includes checking for the absence of training data for MASE, prediction distribution samples for CRPS, quantile predictions for Quantile Loss, and prediction intervals for the Interval Score. |
| **Test Case 4** | The `run_test_unknown_metric` function is designed to test the system's error handling by attempting to calculate a metric with a nonsensical name. This ensures that the evaluator properly identifies and flags unrecognized metric requests with a `ValueError`. |
| **Test Case 5** | The `run_test_length_mismatch` function verifies that the evaluator correctly handles situations where the true and prediction series have different lengths. It confirms that a `ValueError` is raised to prevent erroneous calculations. |
| **Test Case 6** | The `run_test_custom_config` function checks that the `Evaluator` can be correctly initialized with a custom configuration. This test ensures that settings like the list of metrics to calculate and the target column name are properly applied from the configuration object. |
