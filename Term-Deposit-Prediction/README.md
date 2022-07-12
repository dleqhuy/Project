# Term Deposit Prediction
## Giới thiệu

Đây là dataset dùng để dự đoán khả năng khách hàng có muốn gửi tiền dài hạn cho ngân hàng hay không

## Sampling Methods

Bởi vì dữ liệu không cân bằng, nên phải thực hiện việc cân bằng đối với lớp thiểu số mà trong trường hợp này là target __yes__ . Sử dụng kỹ thuật lấy mẫu như SMOTE để có được đầu ra tốt nhất từ các mô hình học máy.
* [__SMOTE__](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
