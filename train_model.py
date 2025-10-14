import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def clean_resume(txt: str) -> str:
    if not isinstance(txt, str):
        return ''
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText.strip()


def main():
    print('Loading dataset...')
    df = pd.read_csv('UpdatedResumeDataSet.csv')
    if 'Resume' not in df.columns or 'Category' not in df.columns:
        raise SystemExit('CSV must contain Category and Resume columns')

    df = df[['Category', 'Resume']].dropna()
    print(f'Rows before dropna: {len(df)}')

    print('Cleaning text...')
    df['clean_resume'] = df['Resume'].astype(str).apply(clean_resume)

    X = df['clean_resume'].values
    y = df['Category'].values

    print('Encoding labels...')
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print('Preparing TF-IDF vectorizer...')
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = tfidf.fit_transform(X)

    print('Training classifier (LogisticRegression)')
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tfidf, y_enc)

    print('Saving tfidf.pkl, clf.pkl, encoder.pkl')
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('clf.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print('Model training complete.')


if __name__ == '__main__':
    main()
