def feature_target_dist(X_train, X_test, y_train, y_test, treshold, criteria):
    """
    :param X_train: Train test split input train set
    :param X_test: Train test split input test set
    :param y_train: Train target set
    :param y_test: Test set
    :param treshold: Feature drop limit as percentage
    :param criteria: ('Mean','Median','Std') Which statistic will be considered for treshold drop
    :return: X_train, X_test
    """
    '''
    Compares feature distributions (mean-median-std) in target and train sets for binary classification models. Useful for 
    final feature elimination on small data sizes where features might not be distributed 
    in a healty way within the two input sets. 
    
    Since calculating Mean and standart deviation not robust to outliers, this function works within 
    %5 - %95 percintiles of the data.
    
    Dependencies: Pandas, Numpy
    '''
    features = X_train.columns.to_list()

    train_means = []
    train_medians = []
    train_stds = []

    test_means = []
    test_medians = []
    test_stds = []

    #Calculating each feature's mean-median-std against target classes
    for i in features:
        train_p_05 = X_train[i].quantile(0.05)
        train_p_95 = X_train[i].quantile(0.95)
        test_p_05 = X_test[i].quantile(0.05)
        test_p_95 = X_test[i].quantile(0.95)

        train_means.append([i, X_train[X_train[i].gt(train_p_05) & X_train[i].lt(train_p_95)].groupby(y_train)[i].mean().to_list()])
        train_medians.append([i, X_train[X_train[i].gt(train_p_05) & X_train[i].lt(train_p_95)].groupby(y_train)[i].mean().to_list()])
        train_stds.append([i, X_train[X_train[i].gt(train_p_05) & X_train[i].lt(train_p_95)].groupby(y_train)[i].mean().to_list()])

        test_means.append([i, X_test[X_test[i].gt(test_p_05) & X_test[i].lt(test_p_95)].groupby(y_test)[i].mean().to_list()])
        test_medians.append([i, X_test[X_test[i].gt(test_p_05) & X_test[i].lt(test_p_95)].groupby(y_test)[i].mean().to_list()])
        test_stds.append([i, X_test[X_test[i].gt(test_p_05) & X_test[i].lt(test_p_95)].groupby(y_test)[i].mean().to_list()])

        mms = pd.DataFrame(train_means, columns = ['Feature', 'Mean'])
        df = pd.DataFrame(test_means, columns = ['Feature', 'Test_Mean'])
        df1 = pd.DataFrame(train_medians, columns = ['Feature', 'Median'])
        df2 = pd.DataFrame(test_medians, columns = ['Feature', 'Test_Median'])
        df3 = pd.DataFrame(train_stds, columns = ['Feature', 'Std'])
        df4 = pd.DataFrame(test_stds, columns = ['Feature', 'Test_Std'])

        mms['Test_Mean'] = df['Test_Mean']

        mms['Median'] = df1['Median']
        mms['Test_Median'] = df2['Test_Median']

        mms['Std'] = df3['Std']
        mms['Test_Std'] = df4['Test_Std']

        del df,df1,df2,df3,df4

        mms[['Train-Mean-0', 'Train-Mean-1']] = pd.DataFrame(mms.Mean.to_list(), index = mms.index)
        mms[['Test-Mean-0', 'Test-Mean-1']] = pd.DataFrame(mms.Test_Mean.to_list(), index=mms.index)

        mms[['Train-Median-0', 'Train-Median-1']] = pd.DataFrame(mms.Median.to_list(), index=mms.index)
        mms[['Test-Median-0', 'Test-Median-1']] = pd.DataFrame(mms.Test_Median.to_list(), index=mms.index)

        mms[['Train-Std-0', 'Train-Std-1']] = pd.DataFrame(mms.Std.to_list(), index=mms.index)
        mms[['Test-Std-0', 'Test-Std-1']] = pd.DataFrame(mms.Test_Std.to_list(), index=mms.index)

        mms.drop(columns = ['Mean','Test_Mean','Median','Test_Median','Std','Test_Std'], axis=1, inplace = True)

        #Calculating stat differences between two sets
        mms['Mean_Diff_0'] = 100 * ((mms['Train-Mean-0'] - mms['Test-Mean-0']) / mms['Train-Mean-0'])
        mms['Mean_Diff_1'] = 100 * ((mms['Train-Mean-1'] - mms['Test-Mean-1']) / mms['Train-Mean-1'])

        mms['Median_Diff_0'] = 100 * ((mms['Train-Median-0'] - mms['Test-Median-0']) / mms['Train-Median-0'])
        mms['Median_Diff_1'] = 100 * ((mms['Train-Median-1'] - mms['Test-Median-1']) / mms['Train-Median-1'])

        mms['Std_Diff_0'] = 100 * ((mms['Train-Std-0'] - mms['Test-Std-0']) / mms['Train-Std-0'])
        mms['Std_Diff_1'] = 100 * ((mms['Train-Std-1'] - mms['Test-Std-1']) / mms['Train-Std-1'])

        #Droping columns with lower stats than treshold in selected criteria
        if criteria == 'Mean':
            l = mms[(np.abs(mms['Mean_Diff_0']) > treshold) | (np.abs(mms['Mean_Diff_1']) > treshold)]['Feature']
            l_list = l.values.tolist()
            X_train.drop(columns = l_list, axis = 1, inplace = True)
            X_test.drop(columns = l_list, axis = 1, inplace = True)
        elif criteria == 'Median':
            l = mms[(np.abs(mms['Median_Diff_0']) > treshold) | (np.abs(mms['Median_Diff_1']) > treshold)]['Feature']
            l_list = l.values.tolist()
            X_train.drop(columns=l_list, axis=1, inplace=True)
            X_test.drop(columns=l_list, axis=1, inplace=True)
        elif criteria == 'Std':
            l = mms[(np.abs(mms['Std_Diff_0']) > treshold) | (np.abs(mms['Std_Diff_1']) > treshold)]['Feature']
            l_list = l.values.tolist()
            X_train.drop(columns=l_list, axis=1, inplace=True)
            X_test.drop(columns=l_list, axis=1, inplace=True)
        else:
            print("Criteria should be: Mean, Median or Std")

        mms

    return X_train, X_test