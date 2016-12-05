# Lịch hoạt động của nhóm học tập Deep Learning (Deep Lunch)

## Buổi offline meeting 7: Recurrent Neural Networks (RNN)

- Thời gian dự kiến: 12:30 ~ 14:00 thứ 5, ngày 15/12/2016

### Chủ đề

- Nắm được những khái niệm cơ bản trong Recurrent Neural Networks (RNN)
- Thực hành RNN với thư viện [DyNet (Dynamic neural network library)](https://github.com/clab/dynet)

### Recommended Readings

- [Chương 10 - Sequence Modeling: Recurrent and Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html)
- [Awesome Recurrent Neural Networks](https://github.com/kjw0612/awesome-rnn), A curated list of resources dedicated to recurrent neural networks.
- [How to Construct Deep Recurrent Neural Networks](https://arxiv.org/abs/1312.6026)
- [Supervised Sequence Labelling with Recurrent Neural Networks](https://pdfs.semanticscholar.org/d145/5ccc018bd42a54d3c0c0f51a4d1963856452.pdf), by Alex Graves.
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Sentence Ordering using Recurrent Neural Networks](https://arxiv.org/abs/1611.02654)
- [Recurrent NNで文書のポジネガ判定する（モデル考案編）](http://olanleed.hatenablog.com/entry/2015/12/07/233307)

### Practice & Softwares

- [Practical Neural Networks for NLP](https://github.com/clab/dynet_tutorial_examples), phần Recurrent Neural Networks.
- [Introduction to Recurrent Networks in TensorFlow](http://danijar.com/introduction-to-recurrent-networks-in-tensorflow/)
- [RNN using LSTM](https://github.com/unnati-xyz/intro-to-deep-learning-for-nlp/blob/master/Recurrent%20Neural%20Networks.ipynb)
- [Fast Recurrent Networks Library](https://github.com/baidu-research/persistent-rnn), by Baidu-research

## Buổi offline meeting 6: Convolutional Neural Network (CNN)

- Thời gian dự kiến: 12:30 ~ 14:00 thứ năm, ngày 8/12/2016

### Chủ đề

- Nắm được những khái niệm cơ bản trong Convolutional Neural Networks
- Sử dụng keras để cài đặt Convolutional Neural Network
- Sử dụng thư viện [DyNet (Dynamic neural network library)](https://github.com/clab/dynet) cho các bài toán cơ bản. Thư viện DyNet trước kia có tên gọi là cnn.

### Recommended Readings

- [Chương 9 - Convolutional Networks](http://www.deeplearningbook.org/contents/convnets.html), sách Deep Learning.
- [Chapter 6, Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html), book "Neural Networks and Deep Learning".
- **Part V - Convolutional Neural Networks**, book "Deep Learning with Python" (đã upload lên github)
- [Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/), course CS231n.
- [Lecture 7 - Convolutional Neural Networks](http://cs231n.stanford.edu/slides/winter1516_lecture7.pdf)
- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/), on WILDML.
- [Gihub repo for class CS231n](https://github.com/cs231n/cs231n.github.io)
- [Visualizing what ConvNets learn](http://cs231n.github.io/understanding-cnn/)
- [UFLDL Tutorial - part Supervised Convolutional Neural Network](http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution)
- [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882), by Yoon Kim.
- [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/), by Chris Olah.
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820)
- [Conv Nets: A Modular Perspective](http://colah.github.io/posts/2014-07-Conv-Nets-Modular)

### Thực hành

- [Handwritten Digit Recognition using Convolutional Neural Networks in Python with Keras](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)
- [Convolutional Neural Networks (LeNet) using Theano](http://deeplearning.net/tutorial/lenet.html)
- [Convolutional Neural Networks](https://www.tensorflow.org/versions/r0.11/tutorials/deep_cnn/index.html), using Tensorflow

## Buổi offline meeting 5: Learning word representations (word2vec algorithm)

- Thời gian dự kiến: 12:30 ~ 14:00 ngày 23/11/2016

### Chủ đề

- Đọc hiểu thuật toán word2vec cho việc học word vector
- Thực hành word2vec trên dữ liệu thật. See [word2vec exercises](https://github.com/minhpqn/DeepLunch/blob/master/docs/word2vec_exercises.md).

### Recommended Readings

- Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. *Distributed Representations of Words and Phrases and their Compositionality*. In Proceedings of NIPS, 2013. [http://arxiv.org/pdf/1310.4546.pdf](http://arxiv.org/pdf/1310.4546.pdf)
- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. *Efficient Estimation of Word Representations in Vector Space*. In Proceedings of Workshop at ICLR, 2013. [http://arxiv.org/pdf/1301.3781.pdf](http://arxiv.org/pdf/1301.3781.pdf)
- [Using Neural Networks for Modelling and Representing Natural Languages](http://www.coling-2014.org/COLING%202014%20Tutorial-fix%20-%20Tomas%20Mikolov.pdf), tutorial at COLING 2014.
- (Youtube) [CS 224D Lecture 2 - Lectures from 2015 ](https://www.youtube.com/watch?v=T8tQZChniMk), by Richard Socher.
- [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
- [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [Making sense of word2vec](https://rare-technologies.com/making-sense-of-word2vec/)
- (Quora) [How does word2vec work?](https://www.quora.com/How-does-word2vec-work)
- Goodman et al., 2001. [A Bit of Progress in Language Modeling](http://research.microsoft.com/en-us/um/redmond/groups/srg/papers/2001-joshuago-tr72.pdf)
- (Blog) [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
- [Semantic Word Vectors and Sentiment Analysis](https://github.com/wellesleynlp/wanyili-finalproject/blob/master/wordvec_sentiment.ipynb)

### Source codes

- [Word2Vec using Theano](https://github.com/mhjabreel/word2vec_theano)
- [word2vec-keras-in-gensim](https://github.com/niitsuma/word2vec-keras-in-gensim)
- [Part 2: Word Vectors](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors), on Kaggle.
- [Vector Representations of Words](https://www.tensorflow.org/versions/r0.11/tutorials/word2vec/index.html), using TensorFlow.
- [Google word2vec](https://github.com/minhpqn/word2vec), cloned from Goolge word2vec code.
- [word2vec in gensim](https://radimrehurek.com/gensim/models/word2vec.html)

## Buổi offline meeting 4: Thuật toán back propagation để huấn luyện mạng Neural

- Thời gian (dự kiến): 12:30 ~ 14:00 ngày 10/11/2016

### Chủ đề

- Học về thuật toán back propagation để huấn luyện mạng Neural (nghe bài giảng của giáo sư Andrew Ng).
- Tìm hiểu cách cài đặt multi-layer perceptron sử dụng Theano và so sánh với các thư viện Keras và scikit-learn.

### Recommended Readings

- [Week 5, Neural Networks: Learning](https://www.coursera.org/learn/machine-learning/home/week/5), by Andrew Ng.
- [Keras: Deep Learning library for Theano and TensorFlow](https://keras.io/)
- [Develop Your First Neural Network in Python With Keras Step-By-Step](http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)
- [Neural network models with scikit-learn](http://scikit-learn.org/stable/modules/neural_networks_supervised.html)

## Buổi offline meeting 3: Feed-forward Neural Networks

- Thời gian (dự kiến): 12:30 ~ 14:00 ngày 2/11/2016

### Chủ đề

- Chủ đề của buổi meeting thứ 3 là mọi người chia sẻ kinh nghiệm khi cài đặt mạng Neural Feed-forward nhiều tầng sử dụng các thư viện Machine Learning phổ biến như: Theano, scikit-learn, tensorflow, cntk, kerras,...
- Ngoài ra chúng ta sẽ cùng nghe các bài giảng về mạng Neural trong các khoá học online và thảo luận các nội dung trong đó.

Các cài đặt sẽ thử nghiệm trên các benchmark data phổ biến:

- [MNIST](http://yann.lecun.com/exdb/mnist/): Dữ liệu nhận dạng chữ viết tay
- Benchmarking data sets cho các thuật toán Deep Learning: [http://deeplearning.net/datasets/](http://deeplearning.net/datasets/)
- Data sets cho bài toán phân lớp: [http://www.is.umk.pl/projects/datasets.html](http://www.is.umk.pl/projects/datasets.html)
- Dữ liệu trong các dự án bạn đã/đang làm.

Mục tiêu:

- Giúp mọi người hiểu được ý tưởng cơ bản của **feed-forward neural network** trước khi đi vào chi tiết khó hơn về thuật toán **backpropagation** dùng để huấn luyện mạng Neural.

### Recommended Readings

- [Chapter 6 - Deep Feedforwad Network](http://www.deeplearningbook.org/contents/mlp.html), Deep Learning book.
- [Week 4, Neural Networks: Representation](https://www.coursera.org/learn/machine-learning/home/week/4), by Andrew Ng.
- [Week 5, Neural Networks: Learning](https://www.coursera.org/learn/machine-learning/home/week/5), by Andrew Ng.
- Video lecture 1.1 ~ 2.11 in [video list](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH) of Hugo Larochelle.
- Chapter 1 and chapter 2, [Neural Network and Deep Learning book](http://neuralnetworksanddeeplearning.com), by Michael Nielsen.
- [Multilayer Perceptron using Theano](http://deeplearning.net/tutorial/mlp.html#mlp)
- [Beginner Tutorial: Neural Nets in Theano](http://outlace.com/Beginner-Tutorial-Theano/)
- [Practical Guide to implementing Neural Networks in Python (using Theano)](https://www.analyticsvidhya.com/blog/2016/04/neural-networks-python-theano/)

## Buổi offline meeting 2: Thuật toán optimization cơ bản, cài đặt thuật toán Perceptron (mạng Neural 1 tầng)

- Thời gian (dự kiến): 12:30 ~ 14:00 ngày 26/10/2016

### Chủ đề:

- Hiểu, cài đặt các thuật toán tối ưu cơ bản
  * Gradient Descent
  * Stochastic Gradident Descent (SGD)
  * Mini-batch gradient descent
- Tìm hiểu, cài đặt thuật toán Perceptron (mạng Neural 1 tầng), thử nghiệm trên các tập dữ liệu thực tế. 
- So sánh các thuật toán trong bài toán phân lớp, thử nghiệm trên các tập dữ liệu
  * Iris data
  * Sentiment classification data
- Sử dụng Theano

### Danh sách tài liệu cần đọc (Readling List)

- [An Introduction to Gradient Descent and Linear Regression](https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/), by Matt Nedrich.
- Learning with large data sets (video lecture), by Andrew Ng: [http://tinyurl.com/jcc4ycv](http://tinyurl.com/jcc4ycv).
- On overview of gradient descent optimization algorithms, by  Sebastian Ruder: [http://tinyurl.com/zf7aqsd](http://tinyurl.com/zf7aqsd).
- Section 5.9, Stochastic Gradient Descent: [http://www.deeplearningbook.org/contents/ml.html](http://www.deeplearningbook.org/contents/ml.html)
- Stochastic Gradient Descent (SGD) library in scikit-learn: [http://scikit-learn.org/stable/modules/sgd.html](http://scikit-learn.org/stable/modules/sgd.html).
- [Theano basic tutorial](http://deeplearning.net/software/theano/tutorial)
- [Deep Learning Tutorials using Theano](http://deeplearning.net/tutorial)

## Buổi offline meeting 1: Cơ bản về Machine Learning và Deep Learning

- Thời gian: 12:30 ~ 14:00 ngày 19/10/2016

### Chủ đề:

- Tại sao cần có nhóm học tập về Deep Learning
- Phương thức hoạt động của nhóm
- Cơ bản về Machine Learning
- Một số kiến thức cơ bản về Deep Learning (Deep Learning Vocabulary)
- Thực hành:
  * Cài đặt các thư viện cho tính toán khoa học trong python
  * Sử dụng scikit-learn cho các bài toán học máy

### Danh sách các tài liệu cần đọc (Reading List)

- Chapter 5: Machine Learning Basics (deep learning book): [http://www.deeplearningbook.org/contents/ml.html](http://www.deeplearningbook.org/contents/ml.html)
- [Lecture notes 1 (Supervised Learning, Discriminative Algorithms)](http://cs229.stanford.edu/notes/cs229-notes1.pdf), cs229 course, Stanford
- [Vocabulary for Deep Learning](https://www.phontron.com/slides/neubig14deeplunch11.pdf), by Graham Neubig
- [Python Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial)
- (Optional) [From Machine Learning to Deep Learning](https://classroom.udacity.com/courses/ud730/lessons/6370362152/concepts/63815621490923) on Udacity

### Bài thực hành (offline meeting 1)

Cài đặt thuật toán Perceptron. Xem [Tutorial của Graham Neubig](http://www.phontron.com/slides/nlp-programming-en-05-perceptron.pdf).















   





