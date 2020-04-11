"""
CIS522-Deep Learning for Data Science: Recurrent Models
Author: Mian Wang  
Time: 4/10/20
"""

# load google drive
from google.colab import drive
drive.mount('/content/drive')

# define the function to plot confusion matrix
from textwrap import wrap
import re
import itertools
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix

# Credits - https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
def plot_confusion_matrix(correct_labels, predict_labels, labels, display_labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
  ''' 
  Parameters:
      correct_labels                  : These are your true classification categories.
      predict_labels                  : These are you predicted classification categories
      labels                          : This is a lit of labels which will be used to display the axis labels
      title='Confusion matrix'        : Title for your matrix
      tensor_name = 'MyFigure/image'  : Name for the output summay tensor
  '''
  cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
  if normalize:
      cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
      cm = np.nan_to_num(cm, copy=True)
      cm = cm.astype('int')
  np.set_printoptions(precision=2)
  fig = matplotlib.pyplot.figure(figsize=(2, 2), dpi=320, facecolor='w', edgecolor='k')
  ax = fig.add_subplot(1, 1, 1)
  im = ax.imshow(cm, cmap='Oranges')

  classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in display_labels]
  classes = ['\n'.join(wrap(l, 40)) for l in classes]
  tick_marks = np.arange(len(classes))
  ax.set_xlabel('Predicted', fontsize=7)
  ax.set_xticks(tick_marks)
  c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()
  ax.set_ylabel('True Label', fontsize=7)
  ax.set_yticks(tick_marks)
  ax.set_yticklabels(classes, fontsize=4, va ='center')
  ax.yaxis.set_label_position('left')
  ax.yaxis.tick_left()

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
  fig.set_tight_layout(True)
  matplotlib.pyplot.show()
  return
  
 
************************************************************* Data Loading and Pre-processing ***************************************************************************
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
 
# import training and validation dataset from google drive
root_path = '/content/drive/My Drive/CIS522 Homeworks/HW4/'
train = pd.read_csv(root_path + 'train.csv')
val = pd.read_csv(root_path + 'val.csv')
    
# import the natural language toolkit to tokenize and lemmatize text
import nltk
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('punkt') # Download this as this allows you to tokenize words in a string.
lemmatizer = WordNetLemmatizer() 

# import 'MPQA_Subjectivity_Lexicon.tff' to determine if a word has a "positive" or a "negative" connotation
with open (root_path+'MPQA_Subjectivity_Lexicon.tff', 'r') as file:
  f = file.readlines()
polarity = {}
for i,s in enumerate(f):
  start1, start2 = s.find('word1=')+6, s.find('priorpolarity=')+14
  len1, len2 = s[start1:].find(' '), s[start2:].find('\n')
  word, pol = s[start1:start1+len1], s[start2:start2+len2]
  lem = lemmatizer.lemmatize(word)
  polarity[lem] = int(pol=='positive')  
  
# define a function to evaluate "ratio" of each review
def ratio(s):
  '''
  compute the ratio of a sentence, i.e. positive/(positive + negative)
  '''
  s = nltk.word_tokenize(s)
  pos, neg = 0, 0
  for w in s:
    lem = lemmatizer.lemmatize(w)
    if lem in polarity: pol = polarity[lem]
    else: pol = 0.5
    if pol==1: pos += 1
    else: neg += 1
  if pos+neg == 0: return 0.5
  else: return pos/(pos+neg), pos, neg
 
# compute the ratio of each review for training set
train['Ratio'], train['pos_neg'] = None, None
for i in range(len(train)):
  ro, pos, neg = ratio(train['Text'][i])
  train['Ratio'][i], train['pos_neg'][i] = ro, (pos, neg)

# compute the ratio of each review for validation set
val['Ratio'], val['pos_neg'] = None, None
for i in range(len(val)):
  ro, pos, neg = ratio(val['Text'][i])
  val['Ratio'][i], val['pos_neg'][i] = ro, (pos, neg)
 
# pick the threshhold for 5 ratings (the baseline threshold to determine whether the rating is 1,2,3,4,5)
ratio_at_score_1 = train[train['Score']==1]['Ratio']
ratio_at_score_2 = train[train['Score']==2]['Ratio']
ratio_at_score_3 = train[train['Score']==3]['Ratio']
ratio_at_score_4 = train[train['Score']==4]['Ratio']
ratio_at_score_5 = train[train['Score']==5]['Ratio']
mean1, mean2, mean3, mean4, mean5 = np.mean(ratio_at_score_1), np.mean(ratio_at_score_2), np.mean(ratio_at_score_3), np.mean(ratio_at_score_4), np.mean(ratio_at_score_5)
thresh12, thresh23, thresh34, thresh45 = np.mean([mean1,mean2]), np.mean([mean2,mean3]), np.mean([mean3,mean4]), np.mean([mean4,mean5])

# define the function to predict rating based on ratio and threshold
def classify(ratio):
  if ratio <= thresh12: return 1
  elif ratio <= thresh23: return 2
  elif ratio <= thresh34: return 3
  elif ratio <= thresh45: return 4
  else: return 5
    
# predict the rating for training set (with baseline threshold)
train['Predict'] = None
for i in range(len(train)):
  train['Predict'][i] = classify(train['Ratio'][i])

# predict the rating for validation set (with baseline threshold)
val['Predict'] = None
for i in range(len(val)):
  val['Predict'][i] = classify(val['Ratio'][i])
  
# calculate F1-score for train and val dataset
F1_train = f1_score(list(train['Score']), list(train['Predict']), average='macro')
F1_val = f1_score(list(val['Score']), list(val['Predict']), average='macro')
print('F1-Score for training set: {:.4f} \nF1-Score for validation set: {:.4f}'.format(F1_train, F1_val))
  
  
*************************************************************** Featuring the Dataset using Torchtext *******************************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import torchtext.data as data
from collections import Counter
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# launch the tensorboard
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
# Load the TensorBoard notebook extension
%load_ext tensorboard

# display the tensorboard
%tensorboard --logdir '/content/runs'


# create torchtext datafield to process the input data so it can later be converted to a tensor
TEXT = data.Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True)
LABEL = data.Field(sequential=False, use_vocab=False)

# create a tabular dataset
datafields = [(' ', None), (' ', None), (' ', None), (' ', None), (' ', None), (' ', None), \
              (' ', None), ('Score', LABEL), (' ', None), ('Summary', TEXT), ('Text', TEXT)]
training_data = data.TabularDataset(root_path + 'train.csv', format='csv', fields=datafields, skip_header=True)
val_data = data.TabularDataset(root_path + 'val.csv', format='csv', fields=datafields, skip_header=True)

# build vocab with GloVe (use the glove.6B.300d word embedding, and save the vocbulary)
TEXT.build_vocab(training_data, val_data, min_freq=3, vectors=torchtext.vocab.GloVe(name='6B', dim=300))

# create an iterator for the dataset
batch_size = 64
train_iterator = data.BucketIterator(
    training_data, 
    batch_size = batch_size, 
    sort_key = lambda x: len(x.Text),
    sort_within_batch = True, 
    repeat = False, 
    shuffle = True,
    device = device ) 
val_iterator = data.BucketIterator(
    val_data,
    batch_size = batch_size,
    sort_key = lambda x: len(x.Text),
    sort_within_batch = True,
    repeat = False,
    shuffle = True,
    device = device )
    

******************************************************************* Recurrent Amazon Reviews Classifier *****************************************************************
class ReviewClassifier(nn.Module):
  """ 
  Parameters: 
  mode (string): Type of recurrent layer being used. Types are ['rnn', 'lstm', 'gru', 'bilstm']
  output_size (int): Size of the last layer for classification (number of classes)
  hidden_size (int): Length of the hidden state vector
  vocab_size (int): Length of the vocab (len(TEXT.vocab))
  embedding_length (int): Dimension of the word embedding vector (dimension when building vocab)
  word_embeddings (Tensor): All of the word embeddings generated (TEXT.vocab.vectors)
  """
  def __init__(self, mode, output_size, hidden_size, vocab_size, embedding_length, word_embeddings):
    super(ReviewClassifier, self).__init__()

    if mode not in ['rnn', 'lstm', 'gru', 'bilstm']:
      raise ValueError("Choose a mode from - rnn / lstm / gru / bilstm")

    self.mode = mode
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_length = embedding_length

    # Embedding Layer
    self.embedding = nn.Embedding(self.vocab_size, self.embedding_length)
    self.embedding.weights = nn.Parameter(word_embeddings, requires_grad=False)

    if self.mode == 'lstm':
      # LSTM Layer
      self.recurrent1 = nn.LSTM(self.embedding_length, self.hidden_size)
      self.recurrent2 = nn.LSTM(self.embedding_length, self.hidden_size)
    elif self.mode == 'bilstm':
      # BILSTM Layer
      self.recurrent1 = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional=True)
      self.recurrent2 = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional=True)
    elif self.mode == 'gru':
      # GRU Layer
      self.recurrent1 = nn.GRU(self.embedding_length, self.hidden_size)
      self.recurrent2 = nn.GRU(self.embedding_length, self.hidden_size)
    else:
      # RNN Layer
      self.recurrent1 = nn.RNN(self.embedding_length, self.hidden_size)
      self.recurrent2 = nn.RNN(self.embedding_length, self.hidden_size)

    # Fully Connected Layer
    self.fc1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
    self.fc2 = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, text, summary, text_lengths, summary_lengths):
    # Embed the reviews
    embedded_text = self.embedding(text)
    embedded_summary = self.embedding(summary)

    # Pack the embedded text
    packed_text = nn.utils.rnn.pack_padded_sequence(embedded_text, text_lengths)
    packed_summary = nn.utils.rnn.pack_padded_sequence(embedded_summary, summary_lengths, enforce_sorted=False)

    # Recurrent layer
    if self.mode == 'rnn' or self.mode == 'gru':
      _, hidden_text = self.recurrent1(packed_text)
      _, hidden_summary = self.recurrent2(packed_summary)
    elif self.mode == 'lstm':
      _, (hidden_text, _) = self.recurrent1(packed_text)
      _, (hidden_summary, _) = self.recurrent2(packed_summary)
    else:
      _, (hidden_text, _) = self.recurrent1(packed_text)      
      _, (hidden_summary, _) = self.recurrent2(packed_summary)
      hidden_text = (hidden_text[0] + hidden_text[1]).unsqueeze(0)
      hidden_summary = (hidden_summary[0] + hidden_summary[1]).unsqueeze(0)

    # Fully connected layer
    fc_input = torch.cat((hidden_text, hidden_summary), dim=2).squeeze(0)
    temp = nn.ReLU()(self.fc1(fc_input))
    prediction = self.fc2(temp)
    return prediction
    
 
# define a function to realize the training process
def train_classifier(model, dataset_iterator, loss_function, optimizer, epochs = 10, log = "runs", verbose = True, print_every = 100, recurrent = False):
  # tensorboard writer
  writer = SummaryWriter(log_dir=log) 
  model.train()
  step = 0

  for epoch in range(epochs):
    total = 0
    total_loss = 0    
    for batch in dataset_iterator:
      text, text_length = batch.Text
      summary, summary_length = batch.Summary
      labels = batch.Score - 1
 
      batch_size = len(labels)
      if torch.sum(text_length)<batch_size or torch.sum(summary_length)<batch_size or any(text_length<=0) or any(summary_length<=0):
        continue
      
      optimizer.zero_grad()
      if recurrent:
        output = model(text, summary, text_length, summary_length).squeeze(1)
      else:
        output = model(text, summary).squeeze(1) 

      labels = labels.type_as(output).long()
      loss = loss_function(output, labels)
      loss.backward()
      optimizer.step()

      _,pred = torch.max(output.data, 1)
      f1 = f1_score(labels.data.cpu(), pred.data.cpu(), average='macro')
      total += labels.size(0)
      total_loss += loss.item()
  
      if ((step % print_every) == 0):
        writer.add_scalar("Training Loss", total_loss/total, step)
        writer.add_scalar("Training F1-Score", f1, step)
        if verbose:
          print("--- Step: %s F1-Score: %.4f Loss: %.4f" %(step, f1, total_loss/total))
      step = step+1
    print("Epoch: %s F1-Score: %.4f Loss: %.4f"%(epoch+1, f1, total_loss/total))


# define a function to realize the validation process
def evaluate_classifier(model, dataset_iterator, loss_function, recurrent = False):
  model.eval()
  total = 0
  total_loss = 0
  true_labels, predict_labels = [], []
  
  for batch in dataset_iterator:
    text, text_length = batch.Text
    summary, summary_length = batch.Summary
    labels = batch.Score - 1

    if any(text_length<=0) or any(summary_length<=0): continue    
    if recurrent:
      output = model(text, summary, text_length, summary_length).squeeze(1)
    else:
      output = model(text, summary).squeeze(1)

    labels = labels.type_as(output).long()
    loss = loss_function(output, labels)
    _,pred = torch.max(output.data, 1)    
    true_labels += (labels+1).tolist()
    predict_labels += (pred+1).tolist()

    f1 = f1_score(labels.data.cpu(), pred.data.cpu(), average='macro')
    total += labels.size(0)
    total_loss += loss.item()
  print("Test statistics: F1-Score: %.4f Loss: %.4f"%(f1, total_loss/total))
  return true_labels, predict_labels

# set the hyper-parameters 
output_size = 5
hidden_size = 256
vocab_size = len(TEXT.vocab)
embedding_length = 300
word_embeddings = TEXT.vocab.vectors
epochs = 5

# Establish LSTM to train/validate dataset and plot the confusion matrix
model_lstm = ReviewClassifier('lstm', output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model_lstm = model_lstm.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

logger_lstm = 'runs/lstm'
train_classifier(model_lstm, train_iterator, criterion, optimizer, log = logger_lstm, epochs = epochs, print_every = 100, recurrent = True)
true_labels_lstm, predict_labels_lstm = evaluate_classifier(model_lstm, val_iterator, criterion, recurrent = True)
plot_confusion_matrix(true_labels_lstm, predict_labels_lstm, [1,2,3,4,5], ['1','2','3','4','5'], title='Confusion matrix', tensor_name = 'LSTM', normalize=False)

# Establish RNN to train/validate dataset and plot the confusion matrix
model_rnn = ReviewClassifier('rnn', output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model_rnn = model_rnn.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_rnn.parameters(), lr=0.001)

logger_rnn = 'runs/rnn'
train_classifier(model_rnn, train_iterator, criterion, optimizer, log = logger_rnn, epochs = epochs, print_every = 100, recurrent = True)
true_labels_rnn, predict_labels_rnn = evaluate_classifier(model_rnn, val_iterator, criterion, recurrent = True)
plot_confusion_matrix(true_labels_rnn, predict_labels_rnn, [1,2,3,4,5], ['1','2','3','4','5'], title='Confusion matrix', tensor_name = 'RNN', normalize=False)

# Establish GRU(Gated Recurrent Unit) to train/validate dataset and plot the confusion matrix
model_gru = ReviewClassifier('gru', output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model_gru = model_gru.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_gru.parameters(), lr=0.001)

logger_gru = 'runs/gru'
train_classifier(model_gru, train_iterator, criterion, optimizer, log = logger_gru, epochs = epochs, print_every = 100, recurrent = True)
true_labels_gru, predict_labels_gru = evaluate_classifier(model_gru, val_iterator, criterion, recurrent = True)
plot_confusion_matrix(true_labels_gru, predict_labels_gru, [1,2,3,4,5], ['1','2','3','4','5'], title='Confusion matrix', tensor_name = 'GRU', normalize=False)

# Establish BILSTM to train/validate dataset and plot the confusion matrix
model_bilstm = ReviewClassifier('bilstm', output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model_bilstm = model_bilstm.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_bilstm.parameters(), lr=0.001)

logger_bilstm = 'runs/bilstm'
train_classifier(model_bilstm, train_iterator, criterion, optimizer, log = logger_bilstm, epochs = epochs, print_every = 100, recurrent = True)
true_labels_bilstm, predict_labels_bilstm = evaluate_classifier(model_bilstm, val_iterator, criterion, recurrent = True)
plot_confusion_matrix(true_labels_bilstm, predict_labels_bilstm, [1,2,3,4,5], ['1','2','3','4','5'], title='Confusion matrix', tensor_name = 'BILSTM', normalize=False)


********************************************************************** Classifier with self-attention *******************************************************************
class Attention(nn.Module):
  def __init__(self, hidden_size):
    super(Attention, self).__init__()
    self.fc1 = nn.Linear(hidden_size, 1)
    
  def forward(self, hidden, encoder_outputs):
    """ 
    Parameters: 
    hidden (vector): Final hidden state from the input sequence
    encoder_outputs (tensor): Hidden state produced from each of the input sequence tokens
    """
    alpha = self.fc1(encoder_outputs)
    alpha = F.softmax(alpha, dim=0)
    context = torch.sum(alpha * encoder_outputs, dim=0).unsqueeze(0)
    att = torch.cat((context, hidden), dim=2)
    return att
attention = Attention(hidden_size).to(device)


class ReviewClassifierWithAttention(nn.Module):  
  def __init__(self, mode, output_size, hidden_size, vocab_size, embedding_length, word_embeddings):
    super(ReviewClassifierWithAttention, self).__init__()

    if mode not in ['rnn', 'lstm', 'gru', 'bilstm']:
      raise ValueError("Choose a mode from - rnn / lstm / gru / bilstm")

    self.mode = mode
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_length = embedding_length

    # Embedding Layer
    self.embedding = nn.Embedding(self.vocab_size, self.embedding_length)
    self.embedding.weights = nn.Parameter(word_embeddings, requires_grad=False)

    if self.mode == 'lstm':
      # LSTM Layer
      self.recurrent1 = nn.LSTM(self.embedding_length, self.hidden_size)
      self.recurrent2 = nn.LSTM(self.embedding_length, self.hidden_size)
    elif self.mode == 'bilstm':
      # BILSTM Layer
      self.recurrent1 = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional=True)
      self.recurrent2 = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional=True)
    elif self.mode == 'gru':
      # GRU Layer
      self.recurrent1 = nn.GRU(self.embedding_length, self.hidden_size)
      self.recurrent2 = nn.GRU(self.embedding_length, self.hidden_size)
    else:
      # RNN Layer
      self.recurrent1 = nn.RNN(self.embedding_length, self.hidden_size)
      self.recurrent2 = nn.RNN(self.embedding_length, self.hidden_size)

    # Fully Connected Layer
    self.fc1 = nn.Linear(4 * self.hidden_size, self.hidden_size)
    self.fc2 = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, text, summary, text_lengths, summary_lengths):
    # Embed the sentences
    embedded_text = self.embedding(text)
    embedded_summary = self.embedding(summary)
   
    # Pack the embedded text
    packed_text = nn.utils.rnn.pack_padded_sequence(embedded_text, text_lengths)
    packed_summary = nn.utils.rnn.pack_padded_sequence(embedded_summary, summary_lengths, enforce_sorted=False)

    # Recurrent layer
    if self.mode == 'rnn' or self.mode == 'gru':
      output_text, hidden_text = self.recurrent1(packed_text)
      output_summary, hidden_summary = self.recurrent2(packed_summary)
    elif self.mode == 'lstm':
      output_text, (hidden_text, _) = self.recurrent1(packed_text)
      output_summary, (hidden_summary, _) = self.recurrent2(packed_summary)
    else:
      output_text, (hidden_text, _) = self.recurrent1(packed_text)      
      output_summary, (hidden_summary, _) = self.recurrent2(packed_summary)
      hidden_text = (hidden_text[0] + hidden_text[1]).unsqueeze(0)
      hidden_summary = (hidden_summary[0] + hidden_summary[1]).unsqueeze(0)

    # Unpack the text
    unpacked_text,_ = nn.utils.rnn.pad_packed_sequence(output_text)    
    unpacked_summary,_ = nn.utils.rnn.pad_packed_sequence(output_summary)
    if self.mode == 'bilstm':
      unpacked_text = unpacked_text[:,:,:self.hidden_size] + unpacked_text[:,:,self.hidden_size:]
      unpacked_summary = unpacked_summary[:,:,:self.hidden_size] + unpacked_summary[:,:,self.hidden_size:]

    # Attention layer    
    attention_text = attention(hidden_text, unpacked_text)
    attention_summary = attention(hidden_summary, unpacked_summary)

    # Fully connected layer
    fc_input = torch.cat((attention_text, attention_summary), dim=2).squeeze(0)
    temp = nn.ReLU()(self.fc1(fc_input))
    prediction = self.fc2(temp)
    return prediction


# Establish BILSTM with "attention" to train/validate dataset and plot the confusion matrix
model_bilstm_att = ReviewClassifierWithAttention('bilstm', output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model_bilstm_att = model_bilstm_att.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_bilstm_att.parameters(), lr=0.001)

logger_bilstm_att = 'runs/bilstm_att'
train_classifier(model_bilstm_att, train_iterator, criterion, optimizer, log = logger_bilstm_att, epochs = epochs, print_every = 100, recurrent = True)
true_labels_bilstm_att, predict_labels_bilstm_att = evaluate_classifier(model_bilstm_att, val_iterator, criterion, recurrent = True)
final_accuracy = np.argwhere(np.array(true_labels_bilstm_att)==np.array(predict_labels_bilstm_att)).size/len(true_labels_bilstm_att)
plot_confusion_matrix(true_labels_bilstm_att, predict_labels_bilstm_att, [1,2,3,4,5], ['1','2','3','4','5'], title='Confusion matrix', tensor_name = 'BILSTM', normalize=False)


**************************************************************** Transfer Learning using Hugging Face *************************************************************************
# install the library 'Simple Transformers', which makes transfer learning extremely easy
!pip install simpletransformers

# initialize the classification model
from simpletransformers.classification import ClassificationModel
model1 = ClassificationModel('roberta', 'roberta-base', num_labels=6, args=({'fp16': False, 'reprocess_input_data': True}))
model2 = ClassificationModel('camembert', 'camembert-base', num_labels=6, args=({'fp16': False, 'reprocess_input_data': True}))

# clean the data with designated form
train_df_cleaned = train[['Text', 'Score']].rename(columns={"Text":"text", "Score":"labels"})
test_df_cleaned = val[['Text', 'Score']].rename(columns={"Text":"text", "Score":"labels"})

# train the model('roberta')
model1.train_model(train_df_cleaned, output_dir='outputs/model1')
# evaluate the model('roberta')
result1, model_outputs1, wrong_predictions1 = model1.eval_model(test_df_cleaned)

# train the model('camembert')
model2.train_model(train_df_cleaned, output_dir='outputs/model2')
# evaluate the model('camembert')
result2, model_outputs2, wrong_predictions2 = model2.eval_model(test_df_cleaned)

# predict with trained model('roberta'), compute F1-score, and plot confusion matrix
predict1 = np.argmax(model_outputs1, axis=1).tolist()
F1_score1 = f1_score(test_df_cleaned['labels'].tolist(), predict1, average='macro')
plot_confusion_matrix(test_df_cleaned['labels'].tolist(), predict1, [1,2,3,4,5], ['1','2','3','4','5'], title='Confusion matrix', tensor_name = 'BILSTM', normalize=False)

# predict with trained model('camembert'), compute F1-score, and plot confusion matrix
predict2 = np.argmax(model_outputs2, axis=1).tolist()
F1_score2 = f1_score(test_df_cleaned['labels'].tolist(), predict2, average='macro')
plot_confusion_matrix(test_df_cleaned['labels'].tolist(), predict2, [1,2,3,4,5], ['1','2','3','4','5'], title='Confusion matrix', tensor_name = 'BILSTM', normalize=False)


************************************************************************* Seq2Seq model ***************************************************************************************
# create torchtext datafield & tabulardataset
TEXT_S2S = data.Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True, init_token='<start>', eos_token='<end>')
datafields = [(' ', None), (' ', None), (' ', None), (' ', None), (' ', None), (' ', None), \
              (' ', None), (' ', None), (' ', None), ('Summary', TEXT_S2S), ('Text', TEXT_S2S)]

train_s2s = data.TabularDataset('/content/drive/My Drive/CIS522 Homeworks/HW4/train.csv', format='csv', fields=datafields, skip_header=True)
val_s2s = data.TabularDataset('/content/drive/My Drive/CIS522 Homeworks/HW4/val.csv', format='csv', fields=datafields, skip_header=True)

# build Vocabulary for dataset
TEXT_S2S.build_vocab(train_s2s, val_s2s, min_freq=3, vectors=torchtext.vocab.GloVe('6B', dim=300))

# establish BucketIterator for training set and validation set
batch_size = 64
train_s2s_iterator = data.BucketIterator(train_s2s, batch_size, sort_key=lambda x: len(x.Text), sort_within_batch=True, shuffle=True, device=device)
val_s2s_iterator = data.BucketIterator(val_s2s, batch_size=1, sort_key=lambda x: len(x.Text), sort_within_batch=True, shuffle=True, device=device)

# set hyper-parameters for seq2seq model
hidden_size = 256
vocab_size = len(TEXT_S2S.vocab)
embedding_size = 300
word_embeddings = TEXT_S2S.vocab.vectors
epochs = 1

# Define a BILSTM Encoder
class Encoder(nn.Module):
  def __init__(self, hidden_size, vocab_size, embedding_size, word_embeddings, dropout_v=0.2):
    super(Encoder,self).__init__()
    self.hidden_size = hidden_size 
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.word_embeddings = word_embeddings
    self.dropout = nn.Dropout(dropout_v)

    # Embedding Layer
    self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
    self.embedding.weights = nn.Parameter(self.word_embeddings, requires_grad=False)

    # BILSTM Layer
    self.recurrent1 = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=True)

  def forward(self, text):
    # Embedding Layer
    embedded_text = self.embedding(text)
    embedded_text = self.dropout(embedded_text)

    # BILSTM Layer
    output_text, (hidden_text,_) = self.recurrent1(embedded_text)
    hidden_text = (hidden_text[0] + hidden_text[1]).unsqueeze(0)
    output_text = output_text[:,:,:self.hidden_size] + output_text[:,:,self.hidden_size:]  
    return output_text, hidden_text
encoder = Encoder(hidden_size, vocab_size, embedding_size, word_embeddings).to(device)


# Define a BILSTM Decoder
class Decoder(nn.Module):
  def __init__(self, hidden_size, vocab_size, embedding_size, word_embeddings, dropout_v=0.2):
    super(Decoder,self).__init__()
    self.hidden_size = hidden_size 
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.word_embeddings = word_embeddings
    self.dropout = nn.Dropout(dropout_v)

    # Embedding Layer
    self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
    self.embedding.weights = nn.Parameter(self.word_embeddings, requires_grad=False)

    # BILSTM Layer
    self.recurrent1 = nn.LSTM(self.embedding_size+self.hidden_size, self.hidden_size, bidirectional=True)

    # Fully Connected Layer
    self.fc1 = nn.Linear(self.hidden_size, self.vocab_size)

  def forward(self, encoder_hidden, decoder_input):    
    # Decoder_input = [batch_size], Embedded_input = [1, batch_size, embedding_size]
    embedded_input = self.embedding(decoder_input.unsqueeze(0))
    embedded_input = self.dropout(embedded_input) 

    # Encoder_hidden = [1, batch_size, hidden_size]
    input_rnn = torch.cat((encoder_hidden, embedded_input), dim=2)
    output_rnn, (hidden_rnn,_) = self.recurrent1(input_rnn)
    output_rnn = output_rnn[:,:,:self.hidden_size] + output_rnn[:,:,self.hidden_size:]

    # Fully Connected Layer
    prediction = self.fc1(output_rnn.squeeze(0))
    return prediction
decoder = Decoder(hidden_size, vocab_size, embedding_size, word_embeddings).to(device)

# put encoder-decoder together
class Encoder_Decoder(nn.Module):
  def __init__(self, encoder, decoder):
    super(Encoder_Decoder,self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, text, summary):
    # set the parameters
    output_size, batch_size, vocab_size = summary.shape[0], summary.shape[1], self.decoder.vocab_size
    outputs = torch.zeros(output_size, batch_size, vocab_size) 
    
    # encoder model
    encoder_output, encoder_hidden = self.encoder(text)
    
    # decoder model
    decoder_input = summary[0,:]
    for t in range(1,output_size):                          
      prediction = self.decoder(encoder_hidden, decoder_input)
      outputs[t] = prediction
      decoder_input = prediction.argmax(1)
    return outputs
model_s2s = Encoder_Decoder(encoder, decoder).to(device)

# define a function for Seq2Seq model training
def train_classifier_s2s(model, dataset_iterator, loss_function, optimizer, batch_size, epochs=10, log="runs", verbose=True, print_every=100):
  # tensorboard writer
  writer = SummaryWriter(log_dir=log) 
  model.train()
  step = 0

  for epoch in range(epochs):
    total = 0
    total_loss = 0    
    for i, batch in enumerate(dataset_iterator):
      text,_ = batch.Text 
      summary,_ = batch.Summary 

      if text.shape[1] < batch_size or summary.shape[1] < batch_size:
        continue
  
      optimizer.zero_grad()
      output = model(text, summary).to(device)
      output_size = output.shape[-1]
      output = output[1:].view(-1, output_size)
      summary = summary[1:].view(-1)

      loss = loss_function(output, summary)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      total += batch_size
  
      if ((step % print_every) == 0):
        writer.add_scalar("Training Loss", total_loss/total, step)
        if verbose:
          print("--- Step: %s Training Loss: %.4f" %(step, total_loss/total))
      step = step+1
      if step == 5000: break
  print("Epoch: %s Training Loss: %.4f"%(epoch+1, total_loss/total))
  

# train data with seq2seq model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_s2s.parameters(), lr=0.001)
logger_s2s = 'runs/s2s'
train_classifier_s2s(model_s2s, train_s2s_iterator, criterion, optimizer, batch_size, epochs=epochs, log=logger_s2s, verbose=True, print_every=100)

def process(l):
  # get rid of '<unk>', '<pad>', '<start>', '<end>'
  stopwords = ['<unk>', '<pad>', '<start>', '<end>']
  new_l = [s for s in l if s not in stopwords]
  res = ' '.join(new_l)
  return res
  
# define a function to evaluate Seq2Seq model
def evaluate_classifier(model, dataset_iterator, log='seqs', batch_size=1):
  # tensorboard writer
  writer = SummaryWriter(log_dir=log) 
  model.eval()

  with torch.no_grad():
    for i,batch in enumerate(dataset_iterator):
      if i == 5: break
      text,_ = batch.Text
      summary,_ = batch.Summary

      if text.shape[1] < batch_size or summary.shape[1] < batch_size:
        continue
      
      output = model(text, summary)
      output_size = output.shape[-1]
      output = output[1:].view(-1, output_size)
      summary = summary[1:].view(-1)
      _,pred = torch.max(output.data, 1) 

      Text = np.array(TEXT_S2S.vocab.itos)[text.view(-1).cpu()].tolist()
      Summary = np.array(TEXT_S2S.vocab.itos)[summary.cpu()].tolist()
      Prediction = np.array(TEXT_S2S.vocab.itos)[pred].tolist()

      print(f'text: {Text}')
      print(f'summary: {Summary}')    
      print(f'predict: {Prediction}')
      writer.add_text('Text', process(Text), i+1)
      writer.add_text('Summary', process(Summary), i+1)
      writer.add_text('Prediction', process(Prediction), i+1)

# evaluate seq2seq model
logger_seqs = 'seqs/s2s'
evaluate_classifier(model_s2s, val_s2s_iterator, logger_seqs)
%tensorboard --logdir '/content/seqs'
