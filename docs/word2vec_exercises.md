# Bài thực hành word2vec

File [enwiki-20150112-400-r10-105752.txt.bz2](http://www.cl.ecei.tohoku.ac.jp/nlp100/data/enwiki-20150112-400-r10-105752.txt.bz2) là file nén dạng bzip2 của 105752 file text được lấy mẫu ngẫu nhiên (tỷ lệ 1/10) từ các bài báo trên Wikipedia có trên 400 từ. Các bài báo trên Wikipedia được lấy vào ngày 12 tháng 1 năm 2015. Sử dụng dữ liệu file này làm corpus để học các vector thể hiện ý nghĩa của các từ.

## 1. Tiền xử lý dữ liệu

Tokenize dữ liệu, lưu file dữ liệu gồm danh sách các từ cách nhau bởi khoảng trắng.

## 2. Xử lý tên các nước tạo thành từ các compound words

Trong tiếng Anh, nhiều từ cạnh nhau có thể tạo thành một từ có ý nghĩa. Ví dụ,
hợp chủng quốc Hoa Kỳ là "United States", vương quốc Anh là "United Kingdom".
Nếu chỉ dùng các "United", "State", hay "Kingdom" như các từ riêng lẻ, ý nghĩa
của các từ này sẽ nhập nhằng. Vì thế trong khi tiền xử lý dữ liệu, ta cần xác
định các từ ghép này. Đoán nhận các từ ghép là một bài toán khó, nên ở đây ta
chỉ đoán nhận các từ ghép là tên của các nước.

Trước hết, download danh sách tên của các nước trên Internet. Dùng danh sách tên
các nước này để đoán nhận các từ ghép là tên nước trong dữ liệu sử dụng ở bài
80, sau đó biến đổi các ký tự spaces thành ký tự underscore (\_) để nối các từ
thành phần. Ví dụ "United States" sẽ trở thành "United\_States", "Isle of Man"
sẽ trở thành "Isle\_of\_Man."

## 3. Sử dụng word2vec để học word vectors

Áp dụng word2vec trên corpus đã tạo ra ở bài 1 để học word vectors

## 4. Hiển thị word vectors

Đọc vào các word vectors trong bài tập 3, hiển thị vector cho từ "United
States". Chú ý là từ "United States" trong corpus đã được biến đổi thành
"United\_States."

## 5. Tính word similarity

Sử dụng word vectors thu được trong bài tập 3, tính cosine similarity cho hai
từ "United States" và "U.S." Chú ý là từ "U.S." trong corpus được lưu trữ là
"U.S"

## 6. Hiển thị top 10 có giá trị similarity cao nhất

Hiển thị top 10 từ với cosine similarity cao nhất với từ "England" và các giá trị cosine similarity tương ứng.

## 7. Các thao tác cộng/trừ word vectors

Đọc vào các word vectors thu được trong bài 85, tính vec("Spain") -
vec("Madrid") + vec("Athens") sau đó hiển thị top 10 từ có cosine similarity gần nhất với vector thu được cùng với các giá trị cosine similarity tương ứng.

## 8. Chuẩn bị dữ liệu analogy

Download dữ liệu [analogy
evaluation](https://github.com/minhpqn/nlp_100_drill_exercises/blob/master/data/questions-words.txt). Trong dữ liệu, các dòng bắt đầu bằng ":" thể hiện tên của section. Ví dụ dòng ":capital-common-countries" bắt đầu cho section "capital-common-countries." Hãy trích xuất các dòng của section "family" trong file đã download và lưu ra file.

## 9. Vận dụng dữ liệu analogy data

Với các dòng trong dữ liệu analogy tạo ra trong bài 8, tính vector sau:
vec(word ở cột 2) - vec(word ở cột 1) + vec(word ở cột 3) sau đó tìm word với
word vector với độ tương tự cao nhất với word vector đã tính đồng thời tính độ
tương tự (cosine similarity). Thêm vào cuối của các dòng từ tìm được và độ tương tự. Trong bài tập này, hãy thử sử dụng word vector đã học được sau bài 3.

## 10. Tính độ chính xác của mô hình trên dữ liệu analogy

Sử dụng dữ liệu của bài 9, tính độ chính xác của các mô hình với mô hình
analogy.

## 11. Tính word similarity trên dữ liệu WordSimilarity-353

Sử dụng đầu vào là dữ liệu [The WordSimilarity-353 Test
Collection](<http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/>),
tính độ tương tự của các từ ở cột 1 và cột 2 và thêm vào cuối các dòng giá trị
độ tương tự này. Hãy áp dụng các mô hình word vectors đã học ở bài 3.

## 12. Đánh giá trên dữ liệu WordSimilarity-353

Sử dụng dữ liệu trong bài 11, sử dụng ranking với các giá trị độ tương tự đã
tính với các mô hình và ranking do con người đưa ra để tính [Spearman
correlation](<https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient>).

## 13. Trích xuất vectors liên quan đến tên nước

Sử dụng mô hình đã học với word2vec, trích xuất các vectors của các từ liên quan đến tên các nước.

## 14. k-means clustering

Lấy đầu vào là các word vectors từ bài tập 13, thực hiện clustering bằng thuật
toán k-means với số lượng clusters *k* = 5.

## 15. Clustering với phương pháp Ward

Lấy đầu vào là các word vectors từ bài tập 13 (các word vectors của tên các
nước), thực hiện hierarchical clustering bằng [phương pháp
Ward](<https://en.wikipedia.org/wiki/Ward's_method>). Sau đó, visualize kết quả
clustering bằng [dendrogram](<https://en.wikipedia.org/wiki/Dendrogram>).

## 16. Visualize word vectors bằng phương pháp t-SNE

Với các word vectors thu được từ bài tập 13, visualize không gian vectors bằng
[phương pháp t-SNE](<http://www.jmlr.org/papers/v9/vandermaaten08a.html>).
