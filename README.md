
# Duplicate Question Pairs

The streamlit app takes two questions as an input and predicts whether The questions are Duplicate or unique.
This is quite useful in providing services to question and answer platform and forums such as Quora,Stack Overflow.Yahoo Answers etc.

The machine learning model was created mainly using:
1)Bag of Words(Tf-idf)
2)Fuzzy features
3)Word2vec
4)Ner tagging



## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


# Hi, I'm Nishant Thakur! 👋


## 🚀 About Me
I am a engineer by qualification who is highly curious about numbers and data.I have a warm enthusiasm for converting data to some useful insights and models.


## Roadmap
I would like to divide this project's roadmap in 5 phases.

1)Understanding the data(EDA)

Some of the insights I found are in the notebook attached in the 
repository.You check them there.

2)Preprocessing the data.

The data needed to be preprocessed as it had some special characters and
contractions(https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions)
I also used Porter stemmer and WordNetLemmatizer but later decided
to drop them.

3)Modeling with only Bag of Words.

I only used tf-idf at the starting to check the accuracy.The accuracy
was found to be around 73%.My main target was to reach 80%.
Machine learning algorithm used was XGBoost

4)Introducing Feature Engineering.

I tried to generate some useful features such as
a)Difference in lengths of these two questions.
b)Mean length of the two questions.
c)Common words/min. word length between the questions
d)Common words/max. word length between the questions
e)Common words/min. word length between the questions
f)Common stop word words/min. stopword length between the questions
g)Common stop word words/max. stopword length between the questions
h)Common token word words/min. token length between the questions
i)Common token word words/max. token length between the questions
j)Is last word equal?(boolean)
k)Is first word equal?(boolean)

By applying these common features an accuracy of 77% was reached

5)Learning about Fuzzy features.

#https://www.analyticsvidhya.com/blog/2021/06/fuzzywuzzy-python-library-interesting-tool-for-nlp-and-text-analytics/

After applying these fuzzy features an accuracy of 78-79% was reached.

6)Implementing word2vec features.
Used a google pretrained model for generating various word2vec features such as
 'cosine_distance',
       'cityblock_distance', 'jaccard_distance', 'canberra_distance',
       'euclidean_distance', 'minkowski_distance', 'braycurtis_distance','common_nouns','common_verbs','common_adj'.

After this an accuracy of 79.4% was reached.

7)The last step to 80.02% accuracy Ner tagging.

Used ner tagging by nltk library to extract some basic features such as common_noun_count,common_verb_count,common_adj_count.Got an accuracy of 80.02%

8)Saving the model and tf-idf and creating a stream lit app for Implementing it.


## 🛠 Skills
Python,NLP,Machine learning.


## Run Locally

Clone the project

1)Open the duplicate_pairs.py and helper.py on 
any Integrated development environment(Pycharm recommended)
2)Download the google pretrained word2vec model
(https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)
3)Unzip it into bin and specify the downloaded folder in the py files
4)Run the duplicate_pairs.py and type the command generated on a local browser to work on the streamlit app.

## Screenshots
![Screenshot (312)](https://user-images.githubusercontent.com/102639991/175779891-fd85086f-c629-4c82-825f-9f8ce2cf9ca2.png)
![Screenshot (313)](https://user-images.githubusercontent.com/102639991/175780664-3033d255-6011-4b28-b219-3c387826a05f.png)
![Screenshot (314)](https://user-images.githubusercontent.com/102639991/175780669-b225889c-3ed1-4258-8161-4e19fc6e8d69.png)
![Screenshot (315)](https://user-images.githubusercontent.com/102639991/175780673-cf3d1710-34ee-4503-8ae8-ca12b9e5708f.png)
![Screenshot (316)](https://user-images.githubusercontent.com/102639991/175780676-8a4a3fc8-8072-4229-92bb-7baec4e7daa2.png)





## Improvements

Even though this model has a 80.02% accuracy it has a lot of FALSE POSITIVES
 for examples it tends to predict sentences such as "What is the capital of India?" and "What is the capital of Pakistan?" as duplicates.

 Some steps that can be taken to further improve the model can be

 1)Implemeting stemming or lemmatizing in the preprocessing part.
 2)Deriving more Ner useful features.
 3)Using a transformer such as BERT instead of Bag of Words.
 4)Using a neural network such as LSTM ( Long -short-term memory)and GRU( Gated Recurrent Unit )