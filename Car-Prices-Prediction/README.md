# 🚙 Car Prices Prediction
## Introduction


Một trong những thách thức của doanh số bán xe hơi là chính là __giá cả__ để đảm bảo rằng rất nhiều người mua xe và có nhu cầu lớn vì giá này. Các yếu tố ảnh hưởng đến giá của xe hơi là __mileage__, __car size__, __manufacturer__. Vì vậy khá khó khăn con người quyết giá, đặc biệt là khi có rất nhiều tính năng này ảnh hưởng đến giá cả. Một trong những giải pháp cho thách thức này là sử dụng __machine learning__ để hiểu những hiểu biết và đưa ra những dự đoán có giá trị tạo ra lợi nhuận cho các công ty.

<h2> Data Source</h2>

Chúng ta sẽ làm việc với một dữ liệu khá lớn có chứa khoảng __10000__ các điểm dữ liệu
https://www.kaggle.com/CooperUnion/cardataset

## Exploratory Data Analysis (EDA)

Dưới đây là một số quan sát được khám phá từ dữ liệu.
* Một số lượng lớn xe hơi là từ nhà sản xuất __'Chevrolet'__ theo sau là __'Ford'__.
* Tổng số xe được sản xuất trong năm __2015__ là cao nhất trong tất cả các năm trong dữ liệu.
* Có nhiều giá trị còn thiếu cho __Danh mục thị trường ('Market Category')__ và một vài giá trị còn thiếu cho các feature __'Engine HP'__ và __'Engine Cylinders' __.
* Giá của nhà sản xuất __'Bugatti'__ rất cao so với các nhà sản xuất xe khác, giá trị __mã lực__ của xe cũng lớn hơn nhiều so với phần còn lại.

## Kết luận

* Mô hình tốt nhất giảm thiểu __mean absolute error (MAE)__ và  __R2_SCORE__ là __Random Forest Regressor__.
* __Phân tích khám phá dữ liệu (EDA)__ cho thấy giá của nhà sản xuất __Bugatti__ cao hơn đáng kể so với các nhà sản xuất khác.