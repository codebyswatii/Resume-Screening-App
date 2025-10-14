import pickle

# Load artifacts
with open('encoder.pkl','rb') as f:
    le = pickle.load(f)
with open('tfidf.pkl','rb') as f:
    tfidf = pickle.load(f)
with open('clf.pkl','rb') as f:
    clf = pickle.load(f)

print('Label encoder classes:', getattr(le, 'classes_', None))

# If classifier has classes_ attribute, print it
print('Classifier classes_:', getattr(clf, 'classes_', None))

samples = [
    'Experienced HR manager with recruitment, onboarding, employee relations, performance management',
    'Machine learning engineer with Python, pandas, scikit-learn, model building and data analysis',
    'Sales professional with experience in B2B sales, client acquisition, revenue growth and targets'
]

X = tfidf.transform(samples)

pred = clf.predict(X)
print('Raw predictions:', pred)

# Try inverse transforming if encoder available
try:
    inv = le.inverse_transform(pred)
    print('Inverse transformed predictions:', inv)
except Exception as e:
    print('Could not inverse transform:', e)

# If classifier supports predict_proba, print probabilities
if hasattr(clf, 'predict_proba'):
    probs = clf.predict_proba(X)
    print('Prediction probabilities (first 3 columns):')
    print(probs)
else:
    print('Classifier has no predict_proba')

# Print top coefficients for class mapped to "HR" if exists
if hasattr(clf, 'coef_'):
    # find index of HR in encoder classes
    classes = list(getattr(le, 'classes_', []))
    if 'HR' in classes:
        idx = classes.index('HR')
        import numpy as np
        top_idx = np.argsort(clf.coef_[idx])[-20:][::-1]
        feature_names = getattr(tfidf, 'get_feature_names_out', None)
        if feature_names is None:
            feature_names = tfidf.get_feature_names()
        print('Top features for HR:')
        print([feature_names[i] for i in top_idx])
    else:
        print('No HR label in encoder classes')
else:
    print('No coef_ on classifier')
