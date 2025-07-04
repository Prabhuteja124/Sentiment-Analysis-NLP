import pytest
import joblib
import os

vectorizer_path=r'D:\Project\models\TfidfVectorizer.pkl'
model_path=r'D:\Project\models\models_training\SVM_best_model.pkl'

corpus=[
        '211/5 TO 558/7 --- WHAT A RECOVERY BY INDIA 🤯',
        "Witness the IMMORTAL story of Rama vs. Ravana 🏹Ramayana.Our Truth. Our History. Filmed for IMAX.From INDIA for a BETTER World.#Ramayana #RamayanaByNamitMalhotra@malhotra_namit MonsterMindCreations",
        "Finally!Not Mythology… but OUR HISTORY 🙏🔥Our Truth. Our History. – these 4 words just hit DIFFERENT! 🛕🇮🇳",
        "What an amazing visual to recreate the Ramayana ❤️",
        "This is going to be a blockbuster. 🔥",
        "LAUGHED SO #BAD AT THIS",
        "Feeling #evil today, gimme something #evil and #bad to do!",
        "🗣️ Morning Xers: What's an innocent word that makes you blush?",
        "If you're missing #BeyondTheGates, which I'm sure you are, check out @thejonlindstrom aka Mr. Joey Armstrong in @lifetimetv's #PrettyHurts. He's #bad there too 😄 #BTG #BeyondTheGatesCBS",
        'Papa is out on a school night #bad',
        "In Kyiv, a young guy is taken to war right off the street.A guy with a broken arm. They hold him right by that arm.One day they will come for you, know that.",
        "Angel of Death missile",
        "🚨Explosions reported from 3 military bases and military sites near Tehran",
        "not funny 😠"
    ]

expected_length=len(corpus)

@pytest.fixture(scope="module")
def load_vectorizer():
    assert os.path.exists(vectorizer_path)
    return joblib.load(vectorizer_path)

@pytest.fixture(scope="module")
def load_model():
    assert os.path.exists(model_path)
    return joblib.load(model_path)

def test_vectorizer_transforms(load_vectorizer):
    transform=load_vectorizer.transform(corpus)
    assert transform.shape[0]==expected_length

def test_model_prediction(load_model,load_vectorizer):
    transformed=load_vectorizer.transform(corpus)
    preds=load_model.predict(transformed)
    assert len(preds)==expected_length
    assert all(pred in [0,1,2] for pred in preds)


@pytest.mark.parametrize("text",corpus)
def test_model(load_model,load_vectorizer,text):
    try:
        transformed=load_vectorizer.transform([text])
        preds=load_model.predict(transformed)
        print(f'Predictions : {preds}')
        assert preds[0] in [0, 1, 2]
    except Exception as e:
        pytest.fail(f"Model's predictions failed : {str(e)}")


