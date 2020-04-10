## Project Title

Natural-Language-Processing  


### Tasks

Given Amazon Reviews Dataset, 4 models are established and compared here (RNN, LSTM, GRU, BILSTM) to predict custormers' rankings from thier reviews, also the best model is implemented and compared with/without attention; On top of this, hugging face pretrained model is introduced for transfer learning; Last but not least, Seq2Seq model is built to predict customers' summaries according to their reviews.  


### Dataset

The training data has 190 Mb, and validation data has 13 Mb, which can be downloaded here:  
https://drive.google.com/file/d/1fdfRa6frQGBIGxDg4zDB1r8eOtZxALw_/view?usp=sharing  
https://drive.google.com/file/d/1pTqDFL9CiTMgHsP_VL4G2CL745DCWa2Y/view?usp=sharing  
https://drive.google.com/file/d/1NPN0XjMOref0T8JSTy1413QxrNnOCQ7F/view?usp=sharing  


### Introduction



### Data Visualization

To begin with, I need to define a non-deep-learning baseline in case deep learning method performs bad or doesn't fit here. To do this, I utilized $$ratio = \frac{positive}{positive + negative}$$ as criterion to evaluate reviews' score. During the process, I need to lemmatize each word, a portion of original words v.s lemmatized words is as follows:  
<div align=center><img src="https://github.com/MianWang123/Information-Retrieval/blob/master/pics/Jaccard%20distance%20of%2010000%20pairs.png"></div>  
After implementing the baseline, I get the confusion matrix below:  
<div align=center><img src="https://github.com/MianWang123/Information-Retrieval/blob/master/pics/Jaccard%20distance%20of%2010000%20pairs.png"></div>  
where y-axis represents the true score given by customers, x-axis is the predicted score based on our non-deep-learning criterion.  
Then, I can start to built the rnn networks(rnn, lstm, gru, bilstm), their confusion matrix are shown respectively below:
<div align=center><img src="https://github.com/MianWang123/Information-Retrieval/blob/master/pics/Jaccard%20distance%20of%2010000%20pairs.png"></div>  
From this matrix, we can see that rnn is the one that performs worst, next is lstm, gru & bilstm stand out in the 4 models based on F1 score, they have similar performance, except that bilstm is a bit better than gru. Therefore, I tried to improve the best model - bilstm with "attention" module, the confusion matrix is below:  
<div align=center><img src="https://github.com/MianWang123/Information-Retrieval/blob/master/pics/probability%20of%20hit.png"></div>  


To make it look clear, I plot the training loss & F1-score graph of rnn, lstm, gru, bilstm, bilstm with "attention" as follows:    
<div align=center><img src="https://github.com/MianWang123/Information-Retrieval/blob/master/pics/probability%20of%20hit.png"></div>  
As comparison, hugging face transfering learning is introduced, 2 pretrained models(roberta, camembert) are utilized to predict the scores, their confusion matrices are shown below:  
<div align=center><img src="https://github.com/MianWang123/Information-Retrieval/blob/master/pics/probability%20of%20hit.png"></div>  
The transferred model performs better than rnn and lstm, but worse than gru and bilstm(without attention), let alone bilstm with attention. To sum up, pre-trained model generalizes well, but does not necessarily outrun our own model. Given specific training data, choose one suitable model may best the transferred model.  

At last, I developed Seq2Seq model to predict the summary for reviews, it's consisted with encoder, decoder, encoder-decoder models. The raw text(reviews) look like:  
<div align=center><img src="https://github.com/MianWang123/Information-Retrieval/blob/master/pics/probability%20of%20hit.png"></div>  
Below are the ground truth summary and our predicted summary:  
<div align=center><img src="https://github.com/MianWang123/Information-Retrieval/blob/master/pics/probability%20of%20hit.png"></div>  



### Acknowledge  
Special thanks to CIS522 course's professor and TAs for providing the data set and guidance
