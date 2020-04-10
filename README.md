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
#### Baseline
To begin with, I need to define a non-deep-learning baseline in case deep learning method performs bad or doesn't fit here. To do this, I utilized <div align=center><img src="http://chart.googleapis.com/chart?cht=tx&chl= ratio = \frac{positive}{positive + negative}" style="border:none;"></div> as criterion to evaluate reviews' score. During the process, I need to lemmatize each word, a portion of original words v.s lemmatized words is as follows:  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/lemmatized_words.png"></div>  
After implementing the baseline, I get the confusion matrix below:  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/baseline_confusion.png"></div>  
where y-axis represents the true score given by customers, x-axis is the predicted score based on our non-deep-learning criterion.  

#### RNN, LSTM, GRU, BILSTM
Then, I can start to built the rnn networks(rnn, lstm, gru, bilstm), their confusion matrices are shown respectively below:
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/rnn_confusion.png"></div>  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/lstm_confusion.png"></div> 
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/gru_confusion.png"></div>  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/bilstm_confusion.png"></div>  

From these graphs, we can see that rnn performs worst, next is lstm, gru & bilstm stand out in the 4 models, they have similar performance, except that bilstm is a bit better than gru. Therefore, I tried to improve the best model - bilstm with "attention" module, the confusion matrix of bilstm with attention is below:  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/bilstm_att_confusion.png"></div>  

To make it clear, I plot the training loss & F1-score graph of rnn, lstm, gru, bilstm, bilstm with "attention" as follows:    
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/training_loss.png"></div>  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/loss_annotation.png"></div>  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/training_loss.png"></div>  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/loss_annotation.png"></div>  

#### Hugging face transfer learning
As comparison, hugging face transfering learning is introduced, 2 pretrained models(roberta, camembert) are utilized to predict the scores, their confusion matrices are shown below:  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/roberta_confusion.png"></div>  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/camembert_confusion.PNG"></div>  

The transferred model performs better than rnn and lstm, but worse than gru and bilstm(without attention), let alone bilstm with attention. To sum up, pre-trained model generalizes well, but does not necessarily outrun our own model. Given specific training data, choose one suitable model may best the transferred model.  

#### Seq2Seq model
At last, I developed Seq2Seq model to predict the summary for reviews, it's consisted with encoder, decoder, encoder-decoder models. The raw text(reviews) look like:  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/text.png"></div>  
Below are the ground truth summary and our predicted summary:  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/summary.png"></div>  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/prediction.png"></div>  


### Acknowledge  
Special thanks to CIS522 course's professor and TAs for providing the data set and guidance
