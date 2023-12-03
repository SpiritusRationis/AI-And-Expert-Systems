import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Загрузка данных
categories = ['comp.windows.x', 'rec.sport.baseball', 'rec.sport.hockey']
remove = ('headers', 'footers', 'quotes')
data = fetch_20newsgroups(subset='all', categories=categories, remove=remove)

# Предварительная обработка данных
nltk.download('stopwords')
stemmer = SnowballStemmer('english')
en_stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление символов, цифр и пунктуации
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Токенизация
    tokens = text.split()
    # Удаление стоп-слов и стемминг
    tokens = [stemmer.stem(word) for word in tokens if word not in en_stopwords]
    return " ".join(tokens)

# Применение предварительной обработки к данным
preprocessed_data = [preprocess_text(text) for text in data.data]

# Разделение на тренировочный и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, data.target, test_size=0.2, random_state=42)

# Создание TF-IDF векторов
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Обучение модели с использованием GridSearchCV
parameters = {'alpha': [0.1, 1.0, 10.0]}
clf = GridSearchCV(MultinomialNB(), parameters, cv=5)
clf.fit(X_train_tfidf, y_train)

print("Лучшие параметры: ", clf.best_params_)

# Предсказание на тестовом наборе
y_pred = clf.predict(X_test_tfidf)

# Оценка результатов
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=data.target_names))
