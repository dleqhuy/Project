# Airbnb Home Prices Prediction 

## Giới thiệu

Chúng ta có thể dự đoán giá nhà với sự trợ giúp của __Machine Learning__ và __Data Science__. Bằng các features như __longitude__ và __latitude__ giúp xác định giá cùng với các features khác như khu phố (neighborhood) và nhu cầu về khu vực.

## Exploratory Data Analysis (EDA)

* Dựa trên kết quả từ Box plot rằng có một vài ngoại lệ trong giá nhà. Do đó, những giá đó phải được loại bỏ để giảm __mean squared error__ hoặc __mean absolute error__ của các mô hình. __Removing outliers__ là một ý tưởng tốt vì không phải tất cả các mô hình đều mạnh mẽ với các ngoại lệ.
* Một phần lớn người dùng từ Airbnb sẵn sàng thuê toàn bộ căn hộ thay vì phòng __private room__ hoặc phòng  __shared room__.
* Một số lượng lớn các ngôi nhà từ __Manhattan__ sau đó là __Brooklyn__.
* Có rất ít ngôi nhà từ  __Staten Island__ so với các thành phố khác.
## Machine Learning Models
