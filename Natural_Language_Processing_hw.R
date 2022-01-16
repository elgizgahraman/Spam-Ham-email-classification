library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
library(h2o)
library(inspectdf)
library(skimr)

data<-fread('emails.csv')
data%>%view()
data%>%glimpse()
data%>%inspect_na()
data%>%dim()

for (x in 1:nrow (data)){data$id[x]=x}
data<-select(data, id, everything())
data$id<-data$id %>% as.character()   

data%>% skim()

set.seed(123)

split <- data$spam %>% sample.split(SplitRatio = 0.8)
train <- data %>% subset(split == T)
test <- data %>% subset(split == F)

train%>%colnames()

it_train <- train$text %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$id,
         progressbar = F) 

vocab <- it_train %>% create_vocabulary()

vocab %>% 
  arrange(desc(term_count)) %>% 
  head(11)
vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)


dtm_train %>% dim()
identical(rownames(dtm_train), train$id)
train%>%colnames()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 1001)
glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

it_test <- test$text %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$id,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)

stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")
vocab <- it_train %>% create_vocabulary(stopwords = stop_words)

pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)
pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(11) 

vectorizer <- pruned_vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()


glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 12,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)

vocab <- it_train %>% create_vocabulary(ngram = c(1L, 2L))

vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5)
bigram_vectorizer <- vocab %>% vocab_vectorizer()


dtm_train <- it_train %>% create_dtm(bigram_vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 15,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")
dtm_test <- it_test %>% create_dtm(bigram_vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)
