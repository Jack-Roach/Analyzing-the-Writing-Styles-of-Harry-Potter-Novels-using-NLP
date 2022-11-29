# Analyzing the Harry Potter Series with Natural Language Processing
##### By Jack Roach

## Background
---
If you're not familiar with the British fantasy series, Harry Potter is a collection of seven books that depicts a world where a subset of humanity has various types of spell-based magical abilities, referring to themselves as witches and wizards. The books focus on the title protagonist, a young orphan named Harry who discovers his magical abilities and is subsequently enrolled in a school for witchcraft and wizardry known as Hogwarts. The books are arguably the most popular series of novels in history based on their record breaking 500 million copies sold. The Harry Potter series has had a significant impact on pop culture overall, proving to be a powerful inspiration for fiction writers. Over 600 thousand works of fan fiction exist from fans inspired by the characters, settings, and events within these books.

## Problem Satement
---
My goal for this project is to use natural language processing to analyze the text body and writing style of all seven books. Through this, I plan to create a model that can correctly classify a body of text to the book it was taken from. Creating an accurate model will mean distinguishing the writing and language styles of each book, which will allow one to compare other bodies of text such as Harry Potter fanfiction to the books. This can be a useful tool for anyone who is trying to match their writing style to that of J.K. Rowling's, or enthusiastic readers who are searching for additional reading that most closely resembles their favorite book.

## Data Analysis
---
For this project, I obtained the Harry Potter books in the form of seven corpora from the data repository and machine learning community website, Kaggle. The corpora came in the form of text files, containing only the body, chapter titles, and page number indicators of each book. The total size of the corpora is a little over 1.16 million words, not including the titles of each chapter.

In order to train a model to classify bodies of Harry Potter text, I divided each book into its chapters, for a total of 199 chapters to use for training and evaluating a classification model. I trained the models on 149 randomly selected chapters and kept the remaining 50 to use as a validation data set. There is a noticeable data inbalance in terms of book size. Since Harry Potter and the Order of the Phoenix (book 5) has the most chapter, it is used to show the baseline accuracy for a null model. If the model predicted every chapter to be from Order of the Phoenix, it would have an accuracy rate of **19.01**%.

|Book|Book Title|Chapters|
|---|---|---|
|1|Harry Potter and the Philosopher's Stone|17|
|2|Harry Potter and the Chamber of Secrets|18|
|3|Harry Potter and the Prisoner of Azkaban|22|
|4|Harry Potter and the Goblet of Fire|37|
|5|Harry Potter and the Order of the Phoenix|38|
|6|Harry Potter and the Half-Blood Prince|30|
|7|Harry Potter and the Deathly Hallows|37|
 
## Modeling
---
In order to use the text data to train a classification model, we need to convert each chapter corpus into a format that our classification algorithms are capable of interpreting. This is where vectorization comes in.
### Vectorization
Vectorization is the process of using natural language processing (NLP) to convert a word or collection of words into a multi-dimensional numerical vector that a model is able to understand. In order to analyze the writing within these books, I utilized a NLP tool known as “Doc2Vec” from the Python Library Gensim. Doc2Vec is a tool that utilizes a continuous bag of words algorithm to convert documents of any size into a numerical multi-dimensional vector whose similarity can be mathematically compared to other vectors. Doc2Vec is a pre-trained neural network which we can feed the tokenized text data from the Harry Potter books in order to train it on the book series's unique vocabulary and literary patterns. After we train it with various paramaters such as vector dimensionality, number of epochs its being trained for, and minimum word frequency (ignoring all words that appear less than X times), we can use another built-in tool from Doc2Vec to use the model to infer a vector from each chapter's body text. Having done this, we now have a dataset of 200 vectors with which to train a classification model.

### Multi-Classification 
The models I used for multi-classification are:
- Logistic Regression
- KNeighbors Classifier
- Decision Tree Classifier
- Bagging Classifier
- Random Forest Classifier
- AdaBoost Classifier
- SVC - Support Vector Classifier
- Extra Trees Classifier (Extremely Random Trees)
- Gaussian Naive Bayes

## Observations
---
J.K. Rowling's writing style is a lot more consistent across all seven books than I expected. A lot of the classification models have a hard time classifying chapters into the books that have fewer chapters, and there are times where a model will ignore multiple books completely when classifying (e.g. only classifying chapters to books 4, 5, and 7).

A disproportionate amount of chapters gets classified as books 4 and 5. While this is partially explainable by the characteristics of our null model, there were many cases where the books with fewer chapters in them did not receive any classifications at all, which I find statistically improbable. From a literary point of view, one potential reason why chapters might be getting misclassifed as books 4 or 5 is because these two books represent a turning point in the series where the overall theme of the series turns darker. As a result, these books might have more mixed themes than the youthful adventures depicted in book 1 or the death-fueled intensity of book 7, which can confuse the Doc2Vec model.

After running many training attempts on Doc2Vec model, I found that the most consistently decent performing model is the KNearest Neighbors classifier. This does not come as a surprise to me in hindsight as KNN utilizes an algorithm that classifies data points based on relative location to others. This is incredibly relevant given that our feature data is in the form of vectors, which by design have a magnitude and direction which makes a location-based classification algorithm well suited to classify them.

As of June 14th, 2022 - The Doc2Vec Model in this Repo vectorizes the text in a way that enables a KNN classifier with a neighbor count paramter of 10 to have an F1 Score of 29% and an accuracy score of 30% on the test data. The F1 Score and Accuracy score for this same fitted model are 30% and 33%, respectively, which suggests that there is very little overfitting occuring. This model continues to have numerous type 2 errors when classifying books 1 and 2, but three chapters were classified correctly in those two categories so those books were not ignored by the model completely.

## Conclusion
---
With the model beating the baseline accuracy by 10%, we can say that this model does have some predicting potential, but it would not be a trustworthy predictor to use for classifying chapters of Harry Potter books at this time, due to its overwhelming tendancy to misclassify text under book 4 and book 5.

### Further Research and Future Ideas
- Clean text data further - Remove page indicators and address typos that could have occured when converting the books to text files.
- Look into more location sensitive classification models like KNN.
- Hyptertune more paramaters when training the Doc2Vec neural network.
- Implement a streamlit app that utlizes the prediction function in modeling.ipynb to classify any inputted text. People could submit fanfiction and see which book it resembles most closely.
- Try different document sizes besides chapters. Sentences, Paragraphs, etc.
- Different MLP tools - There are many more tools for natural language processing that I have not tried yet.


## References
---
### Harry Potter Research
- Obtaining chapter titles for each book: [Fandom](https://harrypotter.fandom.com/wiki/List_of_chapters_in_the_Harry_Potter_books#Harry_Potter_and_the_Philosopher's_Stone_(23_June_1991–20_June_1992))
- Basic info about the book series such as sales: [Wikipedia](https://en.wikipedia.org/wiki/Harry_Potter)
- Utilized search results to get a count for Harry Potter fanfiction: [Fanfiction](https://www.fanfiction.net/book/Harry-Potter/)
- Corpora for all seven Harry Potter Books: [Kaggle](https://www.kaggle.com/datasets/balabaskar/harry-potter-books-corpora-part-1-7)

### NLP Research
- Conceptual Research: [NLP](https://en.wikipedia.org/wiki/Natural_language_processing)
- Gensim Topic Modeling: [Doc2vec Paragraph Embeddings](https://radimrehurek.com/gensim/models/doc2vec.html)
- "Doc2Vec — Computing Similarity between Documents", by Abdul Hafeez Fahad: [Medium](https://medium.com/red-buffer/doc2vec-computing-similarity-between-the-documents-47daf6c828cd)
- "A Gentle Introduction to Doc2Vec", by Gidi Shperber: [Medium](https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
- "Word Embeddings— Fun with Word2Vec and Game of Thrones", by Ravi Kumar: [Medium](https://medium.com/@khulasaandh/word-embeddings-fun-with-word2vec-and-game-of-thrones-ea4c24fcf1b8)

### Code References
- Gensim Installation: [PyPi Gensim](https://pypi.org/project/gensim/)
- "Multi-Class Text Classification with Doc2Vec & Logistic Regression", by Susan Li: [Towards Data Science](https://medium.com/towards-data-science/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4)
- "Text Classification With Word2Vec", by Github user Nadbor [Github Blog](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
