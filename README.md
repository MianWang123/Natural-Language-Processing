## Project Title

Natural-Language-Processing  


### Tasks

Given Amazon Reviews Dataset, 4 models are established and compared here (RNN, LSTM, GRU, BILSTM) to predict custormers' ratings from thier reviews, also the best model is implemented and compared with/without attention; On top of this, hugging face pretrained model is introduced for transfer learning; Last but not least, Seq2Seq model is built to predict customers' summaries according to their reviews.  


### Dataset

The training data has 190 Mb, and validation data has 13 Mb, which can be downloaded here:  
https://drive.google.com/file/d/1fdfRa6frQGBIGxDg4zDB1r8eOtZxALw_/view?usp=sharing  
https://drive.google.com/file/d/1pTqDFL9CiTMgHsP_VL4G2CL745DCWa2Y/view?usp=sharing  
https://drive.google.com/file/d/1NPN0XjMOref0T8JSTy1413QxrNnOCQ7F/view?usp=sharing  


### Introduction
The whole prodecure can be divided into preprocessing the data, building recurrent models, tranfer learning, and establishing sequence to sequence model. Details can be found in the code, the framework is listed below.  
Part 1  
1. Build the Baseline  
  - lemmatize text  
  - evaluate ratio of each review  
  - realize threshold baseline  
2. Featurize the Dataset Using Torchtext   
  - create torchtext data fields    
  - create a tabular dataset  
  - build vocab  
  - create an iterator for the dataset  
3. Establish the Recurrent Models  
  - design an embedding layer   
  - pack the embedded data (optional)    
  - build a rnn/lstm/gru/bilstm layer  
  - add "attention", "teacher forcing", or "beam search" module (optional)  
  - design a fully connected layer  
4. Train/Evaluate the Model  

Part 2  
1. Transfer Learning Using Hugging Face  

Part 3  
1. Develop the Seq2Seq Model  
  - featurize the dataset like before  
  - build the encoder  
  - build the decoder  
  - build encoder-decoder combined model  
  - train/evaluate the model  
  

### Data Visualization
#### Baseline
To begin with, I need to define a non-deep-learning baseline in case deep learning method performs bad or doesn't fit here. To do this, I utilized "ratio" below as criterion to evaluate reviews' ratings.  <div align=center><img src="http://chart.googleapis.com/chart?cht=tx&chl= ratio = \frac{positive}{positive + negative}" style="border:none;"></div>   During the process, I need to lemmatize each word, a portion of original words v.s lemmatized words is as follows:  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/lemmatized_words.png"></div>  
After implementing the baseline, I get the confusion matrix below:  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/baseline_confusion.png" width='320'/></div>  
where y-axis represents the true ratings of customers', x-axis is the predicted ratings based on our non-deep-learning criterion.  

#### RNN, LSTM, GRU, BILSTM
Then, I can start to built the recurrent models(rnn, lstm, gru, bilstm), their confusion matrices are shown respectively below:
<div align=center><figure class="four">
<img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/rnn_confusion.png" width='320'/><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/lstm_confusion.png" width='320'/><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/gru_confusion.png" width='320'/><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/bilstm_confusion.png" width='320'/>
</figure></div>  

From these graphs, we can see that rnn performs worst, next is lstm, gru & bilstm stand out in the 4 models, they have similar performance, except that bilstm is a bit better than gru. Therefore, I tried to improve the best model - bilstm with "attention" module, the confusion matrix of bilstm with attention is below:  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/bilstm_att_confusion.png" width='320'/></div>  

To make it clear, I plot the training loss & F1-score graph of rnn, lstm, gru, bilstm, bilstm with "attention" as follows:     
<div align=center><figure class="four">
<img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/training_loss.png" width='360'/><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/f1_score.png" width='360'/><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/loss_annotation.png" width='320'/><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/f1_annotation.png" width='320'/>
</figure></div>  

#### Hugging face transfer learning
As comparison, hugging face transfering learning is introduced, 2 pretrained models(roberta, camembert) are utilized to predict the ratings, their confusion matrices are shown below:   
<div align=center><figure class="two">
<img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/roberta_confusion.png" width='280'/><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/camembert_confusion.PNG" width='280'/>
</figure></div>  

The transferred model performs better than rnn and lstm, but worse than gru and bilstm(without attention), let alone bilstm with attention. To sum up, pre-trained model generalizes well, but does not necessarily outrun our own model. Given specific training data, choose one suitable model may best the transferred model.  

#### Seq2Seq model
At last, I developed Seq2Seq model to predict the summary for reviews, it's consisted with encoder, decoder, encoder-decoder models. The raw text(reviews) look like:  
<div align=center><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/text.png" width='450'/></div>  
Below are the ground truth summary and our predicted summary:   
<div align=center><figure class="two">
<img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/summary.png" width='430'/><img src="https://github.com/MianWang123/Natural-Language-Processing/blob/master/pics/prediction.png" width='430'/>
</figure></div>  

### Acknowledge  
Special thanks to CIS522 course's professor and TAs for providing the data set and guidance
