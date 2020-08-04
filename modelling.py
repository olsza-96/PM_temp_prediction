#importing desired libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn import  metrics

def data_processing():

    #importing csv data (data is clean so no need for further cleaning - can be seen in Jupyter notebook attached)
    raw_data = pd.read_csv('pmsm_temperature_data.csv')
    print(raw_data.columns)
    #print(raw_data.head(10))
    #print(raw_data.tail(10))

    #feature engineering
    raw_data['coolant^2'] = np.square(raw_data['coolant'])
    raw_data['coolant^3'] = np.power(raw_data['coolant'], 3)

    raw_data['ambient^2'] = np.square(raw_data['ambient'])
    raw_data['ambient^3'] = np.power(raw_data['ambient'], 3)

    raw_data['speed^2'] = np.square(raw_data['motor_speed'])
    raw_data['speed^3'] = np.power(raw_data['motor_speed'], 3)

    to_drop = ['stator_yoke', 'stator_tooth', 'stator_winding']
    raw_data = raw_data.drop(columns=to_drop)

    #data division according to measurement ID
    profiles = raw_data['profile_id'].unique().tolist()

    data_dict = {element: pd.DataFrame for element in profiles}
    for key in data_dict.keys():
        data_dict[key] = raw_data[:][raw_data['profile_id'] == key]

    print('Number of created subsets ' + str(len(profiles)))

    lengths = {}
    for key in data_dict.keys():
        lengths[key] = len(data_dict[key])
        # print('Subset with profile id: '+ str(key))
        # print(print(len(data_dict[key])))

    #barplot of data divided according to measurement ID
    """sns.barplot(list(lengths.keys()), list(lengths.values()))
    plt.xlabel("Profile id", fontsize=12)
    plt.ylabel("Number of occurences", fontsize=12)"""

    #deleting data having less than 10 k occurances
    clean_dict = {k: v for k, v in data_dict.items() if len(v) >= 10000}
    print("Number of datasets containing less than 10 k occurences: " + str(int(len(data_dict)) - int(len(clean_dict))))

    return clean_dict


def test_RFR(data, dataX, dataY, key):
    # randomforest method - ensemble - for most important features

    model = RandomForestRegressor()
    model.fit(dataX, dataY)
    #print(model.feature_importances_)

    """chart = sns.barplot(x=data.drop(columns=['pm']).columns, y=model.feature_importances_)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Best feature scores for the dataset no. ' + str(key))
    plt.xlabel('Feature')
    plt.ylabel('Score')
    plt.show()"""

def modelling(data, key):
    # now preparation for feature selection for one selected dataset
    X = data.drop(columns=['pm']).values
    Y = data['pm'].values

    test_RFR(data, X, Y, key)
    data = data.drop(columns=['ambient', 'coolant', 'motor_speed', 'coolant^2', 'coolant^3', 'ambient^2', 'ambient^3',
                              'speed^2', 'speed^3'])

    data['i_d2'] = np.square(data['i_d'])
    data['u_d2'] = np.square(data['u_d'])

    # splitting dataset again
    #print(data.columns)
    X = data.drop(columns=['pm', 'profile_id']).values
    Y = data['pm'].values

    model = RandomForestRegressor()
    model.fit(X, Y)
    #print(model.feature_importances_)

    #plot_model_importances(data, model.feature_importances_, key)

    # divide input values into datasets for modelling
    X = data.drop(columns=['pm', 'profile_id']).values
    Y = data['pm'].values

    X_train, X_test, y_train, y_test = train_test_split(X, Y)


    #RANDOM FOREST REGRESSION MODELLING
    parameters = {'bootstrap': True,
                  'min_samples_leaf': 3,
                  'n_estimators': 100,
                  'min_samples_split': 15,
                  'max_features': 'sqrt',
                  'max_depth': 10,
                  'max_leaf_nodes': None}
    RF_model = RandomForestRegressor(**parameters)
    RF_model.fit(X_train, y_train)
    y_pred_RF = RF_model.predict(X_test)

    #plot_RFR_results(y_test, y_pred_RF, key)

    # Evaluate errors
    MAE = metrics.mean_absolute_error(y_test, y_pred_RF)
    MSE = metrics.mean_squared_error(y_test, y_pred_RF)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF))
    cvRMSE = RMSE / np.mean(y_test)
    """print("The following performance results are:")
    print("Mean absolute error: " + str(MAE))
    print("Mean squared error: " + str(MSE))
    print("Root mean squared error: " + str(RMSE))
    print("Coefficient of variation of RMSE " + str(cvRMSE))"""

    errors = {'Measurement_id': key,
            'MAE': MAE,
              'MSE': MSE,
              'RMSE': RMSE,
              'cvRMSE': cvRMSE}

    #dataframe = pd.DataFrame([errors])
    #print(dataframe)
    data['predicted'] = RF_model.predict(X) #create prediction for full data entries

    #print("The keys in data after modelling: " +str(data.columns))
    return data, errors

def plot_model_importances(data, y, key ):
    chart = sns.barplot(x = data.drop(columns=['pm', 'profile_id']).columns, y = y)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Best feature scores for the dataset no. ' +str(key))
    plt.xlabel('Feature')
    plt.ylabel('Score')
    plt.show()

def plot_RFR_results(y_test, y_pred_RF, key):

    #plot of results with X entries
    plt.plot(y_test)
    plt.title('Random Forest Regression Results for dataset no. ' +str(key))
    plt.xlabel('Number of entry')
    plt.ylabel('Predicted v Real value')
    plt.plot(y_pred_RF)
    plt.show()

    #regression plot
    plt.scatter(y_test, y_pred_RF)
    plt.title('Regression results for dataset no. ' +str(key))
    plt.xlabel('Real values')
    plt.ylabel('Predicted values')
    plt.show()


if __name__ == '__main__':
    input_data = data_processing()  #returns dictionary of datasets divided according to measurement ID


    errors = {}
    #now iterating for each dataset

    for key in input_data.keys():
        print("Making model prediction for key no. " +str(key))
        input_data[key], errors[key] = modelling(input_data[key], key)


    results = pd.concat([input_data[key] for key in input_data.keys()])
    error = pd.concat([pd.DataFrame([errors[key]]) for key in errors.keys()])
    results.to_csv('modelling_results.csv', index = False)
    error.to_csv('prediction_errors.csv', index = False)