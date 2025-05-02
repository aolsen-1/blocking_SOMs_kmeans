# Investigation of Blocking Regimes Using Self Organizing Maps (SOMs)

Authors: Anna Olsen and Emma Jilek

## I. Introduction and Background

Atmospheric blocking refers to persistent, quasi-stationary high-pressure systems that disrupt the typical westerly flow in midlatitudes (Rex 1950). Dynamically, these blocking regimes have been found to alter the typical propagation of Rossby waves and can cause the upper-tropospheric flow to become stagnant (Lupo and Smith 1995; Nakamura and Huang 2018). These systems often bring static weather conditions for days to weeks, and can lead to severe heatwaves, droughts, and flooding in some regions during the summer months (Chan et al. 2019). Although atmospheric blocking patterns have a large influence on regional climate variability and extremes, the phenomenon is not well studied in the Central US region, and much of the Northern Hemisphere blocking research is focused on the North Atlantic and North Pacific sectors. 

In the Central U.S., blocking episodes are often associated with strong mid-tropospheric anticyclonic flow that suppresses convection, reduces cloud cover through subsidence, and contributes to elevated surface temperatures and prolonged dry spells due to radiative forcing under clear skies (Chan et al. 2019). Regions within our 23–50N and 105–87W domain are particularly vulnerable to the impacts of blocking events, especially during the warm season when soil moisture deficits and atmospheric feedbacks may act to amplify these heat extremes. Several historical droughts and heatwaves, such as those that occurred in 1980, 2012, and 2023, have been linked to persistent blocking patterns that redirect storm tracks and limit precipitation in the region (). Despite their high impact, the representation and predictability of blocks over North America remain challenging due to their nonlinear dynamics, sensitivity to upstream wave activity, and interaction with large-scale teleconnections like ENSO and the MJO (Henderson et al. 2016; Chen et al. 2022).

In order to improve our understanding of the spatial and temporal structure of blocking in this region, we apply a machine learning technique using self-organizing maps (SOMs), which allows us to develop a framework that is objective for classifying atmospheric patterns based on their similarity. Recent work by Thomas et al. (2021)  used SOMs to develop a SOM-based blocking index (SOM-BI) to detect summer blocking events over Europe, and determined that the new index performed better when identifying blocked patterns compared to traditional indices. Their results highlight the potential for SOMs to detect persistent and more subtle features in the atmosphere which may be overlooked by the traditional indicies-- many of which use threshold-based methods. In this study, we use ERA5 reanalysis data and examine the 90-day June–July–August (JJA) periods for twelve anomalous summers (1940, 1954, 1956, 1980, 2002, 2003, 2006, 2007, 2011, 2012, 2020, and 2023) to identify and cluster blocking-like circulation features. By testing several SOM configurations and evaluating their performance, we aim to document the dominant blocking regimes and assess their frequency, structure, and potential relationship to regional drought and heatwave events.

## II. Data and Methods

Our data for this project was obtained from the European Center for Medium-Range Weather Forecasts Reanalysis Version 5 (ERA5), which provides global atmospheric fields at a 0.25° × 0.25° horizontal resolution. We utilized 6-hourly ERA5 data for the variables of 500 hPa geopotential height (z) and potential vorticity (PV), as these fields are frequently used in the identification and analysis of atmospheric blocking patterns (Pelly and Hoskins 2003, Nakamura and Huang 2018). Our spatial domain covers the Central US, bounded between 23°N–50°N latitude and 105°W–87°W longitude, a region where summertime blocking events have been associated with significant hydroclimatic impacts, including drought and heat waves (Hoerling et al. 2014, Liu et al. 2023).

Our analysis focused on JJA periods for twelve years with known anomalous summer drought conditions: 1940, 1954, 1956, 1980, 2002, 2003, 2006, 2007, 2011, 2012, 2020, and 2023. These years were selected based on drought indices to serve as case studies for the evaluation of blocking event frequency and structure. Data preprocessing included temporal subsetting, spatial regridding, and normalization of the geopotential height and PVU fields to prepare them for SOM training.  The input fields were processed using Python libraries, including Xarray, Numpy, Pandas, MiniSom, Sklearn, Scipy, Matplotlib, and Cartopy, and then reshaped into 2D arrays so each field represented a spatial pattern across the spatialdomain.



1.	MUCAPE > 0 J kg−1
2.	ELR   8°C km−1 over a minimum depth of 200 hPa
3.	EML base minimum of 1000 m AGL and below 500 hPa
4.	Higher RH at EML top (compared to base)
5.	ELR < 8°C km−1 below EML base

etc. etc.

## III. Initial Results

Since EMLs are relatively rare events, the full dataset contains substantially fewer EML cases than no EML cases (Fig. 2). Within the 2012-2021 dataset, EMLs are most frequent in 2012, 2013, 2014, and 2018 (Fig. 2). Since the goal of the project is to find meaningful environmental parameters that can distinguish between EML and no EML classes, year will not be used as a predictor in the model. The diurnal distribution of EMLs indicates that EMLs are less common in the afternoon and early evening hours (18 and 0 UTC), likely due to the erosion of the EML by convection (Fig. 2). 

![alt text](images/figure_2.jpg)
> Figure 2. The number of no EML vs. EML instances in the full dataset (left). The distribution of May EMLs per year (center) and per hour (right). 

In addition to the yearly and diurnal distributions, we examine the distributions of additional features associated with EMLs in the dataset (Fig. 3). Consistent with the literature, EMLs are most frequent in the southern half of the Great Plains in spring, roughly south of 40° N latitude. Vertical profiles associated with EMLs have steep lapse rates, relatively low relative humidity, and sufficient vertical wind shear to support deep, moist convection. Due to the presence of the EML’s capping inversion, many EMLs also have moderate to large MUCIN and fairly high 700 mb temperatures. 

etc. etc.

## IV. Summary

This study examines the feasibility of using a random forest classifier to identify the EML in the central CONUS. EMLs are identified in ten years of 6-hourly May data from the ERA5 and converted into binary output. Select variables from ERA5 are used as feature input for the machine learning model. Feature importance is determined during the model development phase. Summary statistics of the final model output are assessed for the testing dataset. Class balance issues with a proportionally smaller number of EMLs compared to non-EMLs in the training and validation data are a limiting factor in model development; however, we anticipate a general assessment of the feasibility of using a random forest classifier in the identification of EMLs in ERA5. By demonstrating the diagnostic capabilities of the machine learning model, we provide a critical tool for severe weather forecasting in the central CONUS.

## V. References

Agard, V., and K. Emanuel, 2017: Clausius–Clapeyron Scaling of Peak CAPE in Continental Convective Storm Environments. *Journal of the Atmospheric Sciences*, **74**, 3043–3054, https://doi.org/10.1175/JAS-D-16-0352.1.

Andrews, M. S., V. A. Gensini, A. M. Haberlie, W. S. Ashley, A. C. Michaelis, and M. Taszarek, 2024: Climatology of the Elevated Mixed Layer over the Contiguous United States and Northern Mexico Using ERA5: 1979–2021. *Journal of Climate*, **37**, 1833-1851,  https://doi.org/10.1175/JCLI-D-23-0517.1.

etc. etc.

# Requirements Document

We identified the following requirements for this project:

| JET01-01  | Load ERA5 Dataset 
|---------|------------| 
| Priority | High |
| Sprint | 1 |
| Assigned To | Emma |
| User Story   | As developers, we need load data from NetCDF4 files to analyse ane preprocess necessary fields. |                                                                                                                                       | 
| Requirements | |
| | 1. Must load 500 hPa PV and Z500 data with correct units using Xarray.|
| | 2. Must load 500 hPa PV and Z500 data with correct months using Xarray.|
| | 3. Must load 500 hPa PV and Z500 data with the correct years using Xarray.|
| Acceptance Criteria | |
| | 1. Dataset opens without error using xarray.|
| | 2. Required variables z, pv are in dataset with expected shapes.|
| | 3. Dimensions include valid_time, latitude, longitude. |
| | 4. Pressure level for both variables = 500 hPa. |
| Unit Test | | 
```
def test_dataset_opens():
    assert isinstance(ds, xr.Dataset)

def test_variable_presence():
    assert 'pv' in ds.data_vars and 'z' in ds.data_vars

def test_coordinate_dimensions():
    for coord in ['valid_time', 'latitude', 'longitude']:
        assert coord in ds.coords

def test_pressure_level_value():
    assert ds.pressure_level.values[0] == 500.0
```


| JET02-01  | Convert units
|---------|------------| 
| Priority | High |
| Sprint | 1 |
| Assigned To | Anna |
| User Story   | I want to convert geopotential to geopotential height and PV to PVU to work with standard units for variables. |                                                                                                                                       | 
| Requirements | |
| | 1. Convert z to meters by dividing by gravity (9.80665).|
| | 2. Convert pv to PVU by multiplying by 1e6.|
| Acceptance Criteria | |
| | 1. New variables z500 and pv_pvu have correct units.|
| | 2. No NaN or infinite values present.|
| Unit Test | | 
```
def test_unit_conversion():
    z500 = ds['z'] / 9.80665
    pv_pvu = ds['pv'].squeeze() * 1e6
    assert np.isfinite(z500.values).all()
```


| JET03-01  | Create engineered input features
|---------|------------| 
| Priority | High |
| Sprint | 1 |
| Assigned To | Anna |
| User Story   | I want to extract PV mean and Z500 gradient, in line with literature, to input meaningful features into SOM. |                                                                                                                                       | 
| Requirements | |
| | 1. Calculate mean PV over lat/lon.|
| | 2. Compute Z500 horizontal gradient magnitude.|
| Acceptance Criteria | |
| | 1. Output vectors are 1D arrays aligned with time.|
| | 2. No missing values.|
| Unit Test | | 
```
N/A
```


| JET04-01  | Normalize engineered input features
|---------|------------| 
| Priority | High |
| Sprint | 1 |
| Assigned To | Anna |
| User Story   | I need to normalize the input features so that the SOM treats all inputs fairly regardless of units. |                                                                                                                                       | 
| Requirements | |
| | 1. Support both z-score and min-max normalization.|
| | 2. Use standard deviation ratio to select the better method.|
| Acceptance Criteria | |
| | 1. Chosen method yields balanced input variances.|
| | 2. Scaled input has zero mean and unit variance (z-score case).|
| Unit Test | | 
```
def normalization_selection(pv_mean, z500_grad_mean):
    # combine into matrix
    X = np.stack([pv_mean, z500_grad_mean], axis=1)

    # min-max scaling
    min_val = np.min(X)
    max_val = np.max(X)
    scale_factor = 100.0 / (max_val - min_val)
    X_minmax = X * scale_factor

    # Z-score scaling
    X_zscore = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # compare standard deviation ratios of scaled variables to find which method is more balanced (closer to 1)
    std_ratio_minmax = np.std(X_minmax[:, 0]) / np.std(X_minmax[:, 1])
    std_ratio_zscore = np.std(X_zscore[:, 0]) / np.std(X_zscore[:, 1])

    # pick method where std ratio is closest to 1
    if abs(std_ratio_minmax - 1) < abs(std_ratio_zscore - 1):
        method_used = 'minmax'
    else:
        method_used = 'zscore'

    return method_used
```


| JET05-01  | Train SOM models
|---------|------------| 
| Priority | High |
| Sprint | 1 |
| Assigned To | Anna |
| User Story   | I want to train SOMs with different hyperparameters to select the most representative model for my data. |                                                                                                                                       | 
| Requirements | |
| | 1. Train SOMs with varied hyperparameters (x, y, sigma, learning_rate).|
| | 2. Save quantization error (QE) and topographical error (TE) to compare.|
| Acceptance Criteria | |
| | 1. Best SOM has lowest QE and TE.|
| | 2. Grid size and input length match.|
| Unit Test | | 
```
N/A
```


| JET06-01  | Assign Best Matching Units (BMUs)
|---------|------------| 
| Priority | High |
| Sprint | 1 |
| Assigned To | Anna |
| User Story   | I need to map the BMUs to assign each day to a SOM node. |                                                                                                                                       | 
| Requirements | |
| | 1. Use som.winner(x) to assign each time step to a BMU.|
| | 2. Save quantization error (QE) and topographical error (TE) to compare.|
| Acceptance Criteria | |
| | 1. Length matches number of time steps.|
| | 2. Output is a list of tuples.|
| Unit Test | | 
```
N/A
```


| JET07-01  | Link SOM nodes to specific days
|---------|------------| 
| Priority | High |
| Sprint | 1 |
| Assigned To | Anna |
| User Story   | As a climatologist, I need to associate BMUs with time to analyze temporal patterns. |                                                                                                                                       | 
| Requirements | |
| | 1. Create DataFrame with valid_time and bmu labels.|
| | 2. Save quantization error (QE) and topographical error (TE) to compare.|
| Acceptance Criteria | |
| | 1. DataFrame rows match time steps.|
| | 2. BMUs formatted as x_y.|
| Unit Test | | 
```
def test_bmu_labeling():
    labels = [f"{bmu[0]}_{bmu[1]}" for bmu in bmus]
    assert all('_' in label for label in labels)
```


| JET08-01  | Count node frequencies
|---------|------------| 
| Priority | High |
| Sprint | 1 |
| Assigned To | Anna |
| User Story   | I want to visualize SOM node activations, so that I can understand which patterns are dominant. |                                                                                                                                       | 
| Requirements | |
| | 1. Count BMUs and plot heatmap.|
| | 2. Save quantization error (QE) and topographical error (TE) to compare.|
| Acceptance Criteria | |
| | 1. Frequency matrix matches SOM grid.|
| | 2. Total count equals number of time steps.|
| Unit Test | | 
```
def test_bmu_labeling():
    labels = [f"{bmu[0]}_{bmu[1]}" for bmu in bmus]
    assert all('_' in label for label in labels)
```


| JET09-01  | Count node frequencies
|---------|------------| 
| Priority | High |
| Sprint | 1 |
| Assigned To | Anna |
| User Story   | I want to estimate a blocking likelihood for each SOM node, so that I can flag potential blocking regimes. |                                                                                                                                       | 
| Requirements | |
| | 1. Combine z-score-normalized PV and Z500 gradient into score.|
| | 2. Select top 10% as blocking-prone.|
| Acceptance Criteria | |
| | 1. blocking_score exists in output.|
| | 2. Returns expected number of blocked nodes.|
| Unit Test | | 
```
def test_blocking_score_computation():
    score = -mean_features_scaled["pv_mean"] - mean_features_scaled["z500_grad_mean"]
    assert not np.any(np.isnan(score))
```


| JET10-01  | Count node frequencies
|---------|------------| 
| Priority | High |
| Sprint | 1 |
| Assigned To | Anna |
| User Story   | I want to estimate a blocking likelihood for each SOM node, so that I can flag potential blocking regimes. |                                                                                                                                       | 
| Requirements | |
| | 1. Combine z-score-normalized PV and Z500 gradient into score.|
| | 2. Select top 10% as blocking-prone.|
| Acceptance Criteria | |
| | 1. blocking_score exists in output.|
| | 2. Returns expected number of blocked nodes.|
| Unit Test | | 
```
def test_blocking_score_computation():
    score = -mean_features_scaled["pv_mean"] - mean_features_scaled["z500_grad_mean"]
    assert not np.any(np.isnan(score))
```
