# Airbnb Home Prices Prediction 

## Giới thiệu

Chúng ta có thể dự đoán giá nhà với sự trợ giúp của __machine Learning__. Bằng các features như __longitude__ và __latitude__ giúp xác định giá cùng với các features khác như khu phố (neighborhood) và nhu cầu về khu vực.

## Metrics

Vấn đề dự đoán giá là một vấn đề __regression__. Do đó, dưới đây là một số metrics cho vấn đề này.
* [__Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
* [__Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

## Exploratory Data Analysis (EDA)

* Dựa trên kết quả từ Box plot rằng có một vài ngoại lệ trong giá nhà. Do đó, những giá đó phải được loại bỏ để giảm __mean squared error__ hoặc __mean absolute error__ của các mô hình. __Removing outliers__ là một ý tưởng tốt vì không phải tất cả các mô hình đều mạnh mẽ với các ngoại lệ.
* Một phần lớn người dùng từ Airbnb sẵn sàng thuê toàn bộ căn hộ thay vì __private room__ hoặc __shared room__.
* Có số lượng lớn các ngôi nhà từ __Manhattan__ sau đó là __Brooklyn__.
* Có rất ít ngôi nhà từ  __Staten Island__ so với các thành phố khác.
## Machine Learning Models

Có nhiều libraries từ __sklearn__ có thể sử dụng cho các dự đoán học máy này. Dưới đây là một số mô hình __machine learning__ có kết quả tốt được sử dụng trong dự đoán giá nhà.
* [__Decision Trees Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
* [__Random Forests Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [__Neural Networks Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
