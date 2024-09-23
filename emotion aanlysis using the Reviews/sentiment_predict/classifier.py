import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
    
    def preprocess_data(self):
        """Preprocess the dataset including sentiment assignment, text cleaning, and tokenization."""
        self.df['Sentiment'] = self.df['Rating'].apply(lambda x: '1' if x > 3 else '0' if x == 3 else '-1')
        self.df = self.df.dropna()
        self.df['Review_Text'] = self.df['Review_Text'].str.lower()
        
        tokenizer = TreebankWordTokenizer()
        self.df['tokenized_reviews'] = self.df['Review_Text'].apply(tokenizer.tokenize)
        
        english_stopwords = stopwords.words('english')
        custom_words_to_retain = ['nor', 'not', "don't"]
        
        def remove_stop_words(words_list):
            return [t for t in words_list if t not in english_stopwords or t in custom_words_to_retain]
        
        self.df['stopwords_removed'] = self.df['tokenized_reviews'].apply(remove_stop_words)
        
        def remove_punct(words_list):
            return [t for t in words_list if t not in string.punctuation]
        
        self.df['punct_removed'] = self.df['stopwords_removed'].apply(remove_punct)
        
        from nltk.stem import WordNetLemmatizer
        lm = WordNetLemmatizer()
        def lemmatize_list(words_list):
            return [lm.lemmatize(t) for t in words_list]
        
        self.df['lemmatized'] = self.df['punct_removed'].apply(lemmatize_list)
        
        def convert_to_string(words_list):
            return " ".join(words_list)
        
        self.df['cleaned'] = self.df['lemmatized'].apply(convert_to_string)
        self.df.to_csv('Cleaned_Dataset.csv', index=False)
    
    def plot_distributions(self):
        """Plot distributions of ratings and sentiments."""
        plt.figure(figsize=(10, 5))
        sns.countplot(data=self.df, x='Rating', palette='viridis')
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.countplot(data=self.df, x='Sentiment', palette='viridis')
        plt.title('Distribution of Sentiments')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.savefig('ratings.jpg')
    
    def generate_wordcloud(self):
        """Generate and save a word cloud from the cleaned reviews."""
        all_reviews = ' '.join(self.df['cleaned'])
        plt.figure(figsize=(10, 7))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Reviews')
        plt.savefig('wordcloud.jpg')
    
    def train_and_evaluate(self):
        """Train models and evaluate their performance."""
        X = self.df['cleaned'].values
        y = self.df['Sentiment'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_rf.fit(X_train_tfidf, y_train)
        y_pred_rf = clf_rf.predict(X_test_tfidf)
        
        clf_svm = SVC(kernel='linear', random_state=42)
        clf_svm.fit(X_train_tfidf, y_train)
        y_pred_svm = clf_svm.predict(X_test_tfidf)
        
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        report_rf = classification_report(y_test, y_pred_rf)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        report_svm = classification_report(y_test, y_pred_svm)
        
        print(f"Random Forest Classifier Model Accuracy: {accuracy_rf}")
        print("Random Forest Classifier Model Report:")
        print(report_rf)
        
        print(f"Model Accuracy with SVM: {accuracy_svm}")
        print("Classification Report with SVM:")
        print(report_svm)
        
        cm_rf = confusion_matrix(y_test, y_pred_rf, labels=['-1', '0', '1'])
        disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Negative', 'Neutral', 'Positive'])
        plt.figure(figsize=(10, 7))
        disp_rf.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Random Forest')
        plt.show()
        
        cm_svm = confusion_matrix(y_test, y_pred_svm, labels=['-1', '0', '1'])
        disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Negative', 'Neutral', 'Positive'])
        plt.figure(figsize=(10, 7))
        disp_svm.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - SVM')
        plt.savefig('confusion_matrix.jpg')

if __name__ == '__main__':
    file_path = 'healthcare_reviews.csv'
    analysis = SentimentAnalysis(file_path)
    analysis.preprocess_data()
    analysis.plot_distributions()
    analysis.generate_wordcloud()
    analysis.train_and_evaluate()
