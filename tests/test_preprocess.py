import pytest
from src.data_preprocessing.clean_text import TextCleaner

cleanner=TextCleaner()

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

@pytest.mark.parametrize("text",corpus)
def test_preprocess_clean_text(text):
    clean_text=cleanner.preprocess_text(text)
    assert isinstance(clean_text, str)
    assert "!" not in clean_text
    assert "#" not in clean_text
    assert "..." not in clean_text
    print(f'Cleaned Text :{clean_text}')

@pytest.mark.parametrize("text",corpus)
def test_preprocess_lowercase(text):
    cleantext=cleanner.preprocess_text(text)
    assert cleantext==cleantext.lower()
    print(f'Cleaned Text:{cleantext}')

@pytest.mark.parametrize("text",corpus)
def test_preprocess_empty_string(text):
    cleaned_text = cleanner.preprocess_text(text)
    assert cleaned_text!=""
    print(f'Pass {cleaned_text}')

