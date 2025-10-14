import pickle
import PyPDF2

# Load artifacts
with open('encoder.pkl','rb') as f:
    le = pickle.load(f)
with open('tfidf.pkl','rb') as f:
    tfidf = pickle.load(f)
with open('clf.pkl','rb') as f:
    clf = pickle.load(f)

files = ['health_fitness_resume.pdf','NetworkSecurityEng_Resume.pdf']

for fn in files:
    print('---', fn)
    try:
        with open(fn, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    text += t + '\n'
        print('Extracted length:', len(text))
        if len(text) < 20:
            print('Warning: very short extracted text')
        X = tfidf.transform([text])
        pred = clf.predict(X)
        try:
            print('Prediction:', le.inverse_transform(pred)[0])
        except Exception as e:
            print('Could not inverse transform:', e)
    except Exception as e:
        print('Error reading file:', e)
