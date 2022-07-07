# Telecom Customer Churn Prediction


## Giới thiệu

__Telephone__ company, còn được gọi là __telco__, là nhà cung cấp dịch vụ viễn thông cung cấp các dịch vụ như __telephony__ và __data communications access__. Họ chịu trách nhiệm cung cấp các dịch vụ điện thoại cho những người ở các vùng khác nhau của __The United States of America__. Có rất nhiều khách hàng sử dụng dịch vụ truyền thông của Telco. Có những khách hàng cũng chọn các dịch vụ khác như __TV Streaming__ và __movies streaming__. Hiện tại, Telco không thể dự đoán chính xác liệu một khách hàng nhất định đăng ký dịch vụ có sẵn sàng rời bỏ dịch vụ hay không. Nếu họ có thể biết với độ chính xác tốt, họ sẽ có thể đưa ra các kế hoạch và dịch vụ cho những người dùng sẵn sàng rời bỏ dịch vụ đó.

## Machine Learning và Data Science 

Vì __Telco__ có rất nhiều khách hàng __subscribe__ (đăng ký) dịch vụ của họ, nên sẽ rất hữu ích nếu chúng ta có thể dự đoán liệu __customer__ có __churn__ (rời khỏi dịch vụ) trong khoảng thời gian vài ngày hay không. Hơn nữa, chúng ta có thể xem xét các yếu tố __influential__ (ảnh hưởng) đối với khách hàng chẳng hạn như __type of billing__ (loại thanh toán), __age__ và liệu họ có __partner__ hay không. Sau khi xem xét các yếu tố này và nhiều yếu tố khác ảnh hưởng đến sự rời bỏ của khách hàng, công ty có thể đưa ra các kế hoạch đảm bảo khách hàng không rời bỏ dịch vụ của họ.

Project này sử dụng các thuật toán như LogisticRegression, DecisionTreeClassifier, NaiveBayes, RandomForest để phân loại dữ liệu khách hàng có rời bỏ dịch vụ của công ty hay không

## Exploratory Data Analysis (EDA)

* Dựa trên __exploratory data analysis (EDA)__, ta thấy rằng phí hàng tháng của khách hàng có mối tương quan cao với việc khách hàng có chọn kết nối cáp quang hay không.
* Một tỷ lệ lớn khách hàng đã chọn hợp đồng __month-to-month__ thay vì hợp đồng __year-long__ hoặc __two-year__.
* Các khoản phí hàng tháng tương quan với việc một người có phải là __senior__ hay không. Do đó, điều này cung cấp cho chúng ta  cái nhìn sâu sắc rằng những senior có khả năng sẽ đăng ký vào các dịch vụ khác như dịch vụ phát trực tuyến __movies__ và dịch vụ __internet__.
* Dựa trên các biểu đồ,  ta thấy rằng các (device protection plans) kế hoạch bảo vệ thiết bị cũng dẫn đến việc tăng đáng kể phí hàng tháng.

## Metrics sử dụng


Vì biến đầu ra là 0 hoặc 1, nó là một vấn đề __phân loại nhị phân__ với các khả năng là liệu khách hàng sẽ __churn__ hay __not churn__. Do đó, các chỉ số được xem xét cho vấn đề __classification__ như sau.
* [__Accuracy__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
* [__Logistic Loss__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
* [__Precision__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
* [__Recall__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
* [__F1 Score__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
* [__Confusion Matrix__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
