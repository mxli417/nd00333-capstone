# Azure Machine Learning Engineer Capstone Project - machine learning in a heartbeat

This is the Azure Machine Learning Engineer Capstone Project, concluding the coursework. The project consists of
two distinct parts, centered around the prediction of (non-) survival of heart failure patients using data sourced from kaggle. 
The first part comprises a so-to-speak "manual" machine learning project approach, using the Azure ML framework and HyperDrive
to tune the parameters of a Gradient Boosting Classifier, selecting the best model. The second part consists of an AutoML
approach to the problem, using Auto ML framework provided by Azure. We then finally select, deploy, test and consume the 
endpoint of the best model from both approaches. 


## Project Set Up and Installation
This project requires the creation of a compute cluster in order to run the machine learning
experiments. Additionally, the dataset has to be uploaded and ingested manually. 

## Dataset

### Overview
For this project we use the [heart failure prediction dataset from Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data). This dataset contains records of medical data from 299 patients with heart failure, along with the (non-) survival of patients as a binary variable (`DEATH_EVENT`). 

### Task
The main task is to predict the (non-) survival of patients, using the dataset. The target variable or dependent variable is the binary `DEATH_EVENT` variable. The clinical features included in this dataset are: age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, 
sex, smoking and time.

### Access
We manually downloaded and ingested the data from kaggle into Azure ML Studio, uploading
the csv and registering it as an Azure ML dataset, which then can be used via jupyter
notebook, or else. We first and foremost used access via jupyter notebook.

## Automated ML
The AutoML-approach aims towards automating the complete hand-crafted and thus brittle pipeline often used with hand-crafted ML-experiments. It includes data ingestion, feature engineering/learning, model training and hyperparameter selection all in one go. The used AutoML configuration considers the following aspects: 

- the `task` which is set to `classification`
- the `primary metric` used, which is set to `accuracy` according to our task
- the `training_data`, which is simply the Azure ML data asset described above
- the `label_column_name` of the target variable, which is the `DEATH_EVENT` column in the dataset
- the `n_cross_validations` is the number of cross-validation folds used to train and evaluate the models to get a better overview of the individual model's performances
- the `compute_target` to run our Auto ML job on, as described above
- the `enable_early_stopping`-flag to enable early stopping s.t. the experiment ends if the results do not improve satisfactorily
- the `experiment_timeout_minutes` - set to 20 minutes to enable us to act fast if the experiment gets stuck
- the `max_concurrent_iterations`, we allow 5 in order to not overload our compute
- we set up a `path` for storing the experiment results
- we set the `featurization` flag to "auto" to make full use of the AutoML - techniques for (pre-) processing our data
- and finally, we setup up a `debug_log` name for the Auto ML loggings


### Results
The best Auto ML experiment model is a Voting Ensemble, which consists of several scaling steps ad LightGBMClassifier, an XGBoostClassifier and a Random Forest classifier. Here are the details on the ensemble:

```
Overview over the best model and its details: Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(enable_dnn=False, enable_feature_sweeping=True, feature_sweeping_config={}, feature_sweeping_timeout=86400, featurization_config=None, force_text_dnn=False, is_cross_validation=True, is_onnx_compatible=False, observer=None, task='classification', working_dir='/mnt/batch/tasks/shared/LS_root/mount...
                 PreFittedSoftVotingClassifier(classification_labels=array([0, 1]), estimators=[('65', Pipeline(memory=None, steps=[('sparsenormalizer', Normalizer(copy=True, norm='l2')), ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=0.9, eta=0.3, gamma=0, max_depth=6, max_leaves=0, n_estimators=50, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=0, reg_alpha=1.7708333333333335, reg_lambda=1.7708333333333335, subsample=0.9, tree_method='auto'))], verbose=False)), ('48', Pipeline(memory=None, steps=[('sparsenormalizer', Normalizer(copy=True, norm='l2')), ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=0.6, eta=0.5, gamma=0.01, max_depth=6, max_leaves=0, n_estimators=50, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=0, reg_alpha=0, reg_lambda=0.3125, subsample=1, tree_method='auto'))], verbose=False)), ('60', Pipeline(memory=None, steps=[('sparsenormalizer', Normalizer(copy=True, norm='l1')), ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=1, eta=0.4, gamma=0, max_depth=9, max_leaves=255, n_estimators=100, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=0, reg_alpha=1.9791666666666667, reg_lambda=0.625, subsample=0.5, tree_method='auto'))], verbose=False)), ('64', Pipeline(memory=None, steps=[('sparsenormalizer', Normalizer(copy=True, norm='l1')), ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=0.8, eta=0.3, gamma=0, max_depth=6, max_leaves=31, n_estimators=100, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=0, reg_alpha=0.8333333333333334, reg_lambda=0, subsample=0.6, tree_method='auto'))], verbose=False)), ('39', Pipeline(memory=None, steps=[('sparsenormalizer', Normalizer(copy=True, norm='l1')), ('lightgbmclassifier', LightGBMClassifier(boosting_type='gbdt', colsample_bytree=0.7922222222222222, learning_rate=0.03158578947368421, max_bin=140, max_depth=7, min_child_weight=4, min_data_in_leaf=0.013801724137931036, min_split_gain=0.3157894736842105, n_estimators=200, n_jobs=1, num_leaves=71, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=None, reg_alpha=0.7368421052631579, reg_lambda=0.5789473684210527, subsample=0.6931578947368422))], verbose=False)), ('0', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True)), ('lightgbmclassifier', LightGBMClassifier(min_data_in_leaf=20, n_jobs=1, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=None))], verbose=False)), ('72', Pipeline(memory=None, steps=[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('lightgbmclassifier', LightGBMClassifier(boosting_type='goss', colsample_bytree=0.3966666666666666, learning_rate=0.07368684210526316, max_bin=300, max_depth=3, min_child_weight=5, min_data_in_leaf=0.041385172413793116, min_split_gain=0.3157894736842105, n_estimators=50, n_jobs=1, num_leaves=239, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=None, reg_alpha=0.05263157894736842, reg_lambda=0.8421052631578947, subsample=1))], verbose=False)), ('81', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=True, with_std=True)), ('randomforestclassifier', RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None, criterion='gini', max_depth=None, max_features=0.5, max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=0.01, min_samples_split=0.056842105263157895, min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1, oob_score=True, random_state=None, verbose=0, warm_start=False))], verbose=False)), ('76', Pipeline(memory=None, steps=[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('randomforestclassifier', RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None, criterion='gini', max_depth=None, max_features=0.8, max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=0.035789473684210524, min_samples_split=0.01, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))], verbose=False))], flatten_transform=None, weights=[0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111]))],
         verbose=False)
```

The model has an `accuracy` of ~0.89% 

Interestingly, the Auto ML experiment did not reach an accuracy of >90 %.

### Ideas for improvement

Hence, it might be interesting for the future - especially with such a small dataset - to investigate Auto ML [data synthetization approaches](https://ydata.ai/resources/top-5-packages-python-synthetic-data), which could
boost the accuracy and increase the amount of data to learn form. 
Additionally, a different metric, maybe taking into account how costly it is to misclassify the patients, might be more suitable for the task at hand and provide additional insights into the models' performance.

Run Details widget output:
![Run Details Auto ML](../screenshots/AutoML_RunDetails.png)

Best Run ID and metrics (1-2): 
![Best Auto ML model Run ID](../screenshots/AutoML_BestModelID_metrics_v1.png)
![Best Auto ML model metrics](../screenshots/AutoML_BestModelID_metrics_v2.png)

Best model details w. params (also check the complete printout above):
![Best model with params](../screenshots/AutoML_BestModel_Params.png)

Best model registration (1-2):
![Best model registration (1)](../screenshots/AutoML_BestModelRegister.png)
![Best model registration (2)](../screenshots/AutoML_BestModelRegister_v2.png)

## Hyperparameter Tuning
We use a Gradient Boosting Classifier here, because we are dealing with tabular data and have a binary variable as the target in our classification problem. This is an additive modeling approach, often providing very good performance, a lot of flexibility, can work with categorical and numerical values as-is and naturally handles missing data.

To tune and adapt the basic classifier, we are using HyperDrive to select the best hyperparameters which are here:

- the `learning_rate` (default: 0.1, we vary this by allowing `uniform(0.1, 0.5)` as a parameterization option)
- `n_estimators`, the number of base estimators (decision trees) used in the gradient boosting modeling process (we give a list to choose from: 100, 200, 300, 350)

We use a bandit early stopping policy, which halts the experiments if there is no more improvement in model accuracy, i.e. the model primary metric of the last run is no within the specified slack factor of the most successful run.

To progress through the hyperparameter search space (defined on n_estimators and the learning_rate) fast and easy, we use a random parameter sampler. This is bc. of its non-exhaustive nature, sampling suitable hyperparameters randomly.

To best track our experimentation success, we optimize for the best-possible `accuracy` (primary metric) w.r.t. the classification problem, to most accurately predict the (non-) survival of patient's based on their clinical data.

### Results
The best Gradient Boosting Classifier also reaches an accuracy of ~95% with its hyperparameters `learning_rate`=0.1475654756162574 and `n_estimators`=100. 

### Ideas for improvement
Although the accuracy seems competitive w.r.t. the Auto ML approach, it might be beneficial to provide other `subsample` - parameter values, which would allow the model's individual learner's to choose different samples from the dataset during training. Besides, a different metric, maybe taking into account how costly it is to misclassify the patients might be more suitable for the task at hand and provide additional insights into the models' performance.

Hyperdrive Run Details output:
![Run Details HyperDrive](../screenshots/HyperDrive_RunDetails.png)

Hyperdrive model best run id and metrics as well as the best model parameters:
![Hyperdrive best model best run id](../screenshots/HyperDrive_BestModelID_metrics.png)

Hyperdrive best model registration:
![Hyperdrive best model registration](../screenshots/HyperDrive_RegisterBestModel.png)


## Model Deployment
We deployed the best model from the HyperDrive experiment as a web service endpoint and tested it with three randomly chosen samples from the dataset.


```
[{'age': 90.0,
  'anaemia': 1,
  'creatinine_phosphokinase': 47,
  'diabetes': 0,
  'ejection_fraction': 40,
  'high_blood_pressure': 1,
  'platelets': 204000.0,
  'serum_creatinine': 2.1,
  'serum_sodium': 132,
  'sex': 1,
  'smoking': 1,
  'time': 8},
 {'age': 73.0,
  'anaemia': 1,
  'creatinine_phosphokinase': 231,
  'diabetes': 1,
  'ejection_fraction': 30,
  'high_blood_pressure': 0,
  'platelets': 160000.0,
  'serum_creatinine': 1.18,
  'serum_sodium': 142,
  'sex': 1,
  'smoking': 1,
  'time': 180},
 {'age': 50.0,
  'anaemia': 0,
  'creatinine_phosphokinase': 582,
  'diabetes': 0,
  'ejection_fraction': 62,
  'high_blood_pressure': 1,
  'platelets': 147000.0,
  'serum_creatinine': 0.8,
  'serum_sodium': 140,
  'sex': 1,
  'smoking': 1,
```
Screenshot of the successful best model registration:
![Hyperdrive model deployment registration](../screenshots/Service_ModelRegistering.png)

Screenshot of the successful deployment loggings:
![Hyperdrive model deployment loggings](../screenshots/Service_deployment_success_log.png)

Screenshot of the deployed model test:
![Hyperdrive model deployment test](../screenshots/Service_TestEndpoint.png)

Screenshot of the healthy endpoint: 
![Service healthy endpoint](../screenshots/Service_Healthy_Endpoint.png)

## Screen Recording
https://youtu.be/co18TC7lnLk

## Additionl Screenshots

Feel free to check out the additional screenshots from the last submission under: [previous screenshots](../leg_screen/)