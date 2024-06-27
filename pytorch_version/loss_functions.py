def mse(model, test_data, anomaly_data): 
    reconstructed_anomaly = model.predict(anomaly_data) #pass known anomaly data into model
    model.anomaly_scores = np.mean(np.square(anomaly_data - reconstructed_anomaly), axis=(1,2,3)) #compute the anomaly scores

    #pass known background data into the model
    reconstructed_test = model.predict(test_data)
    model.test_scores = np.mean(np.square(test_data - reconstructed_test), axis=(1,2,3))

    #print mse over anomolous and non anomolus data
    print('anomaly MSE (loss) over all anomalous inputs: ', np.mean(model.anomaly_scores)) 
    print('not anomaly MSE (loss) over all non-anomalous inputs: ', np.mean( model.test_scores))
