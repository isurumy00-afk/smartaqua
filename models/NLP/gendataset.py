import random
import re
import pandas as pd

# ============================================================
# Realistic Noisy Fish Disease NLP Dataset Generator
# Output: fish_symptom_dataset.csv
# ============================================================

random.seed(42)

OUTPUT_FILE = "fish_symptom_dataset.csv"

SAMPLES_PER_CLASS = 600

disease_labels = {
    0: "Bacterial Red disease",
    1: "Bacterial diseases - Aeromoniasis",
    2: "Bacterial gill disease",
    3: "Fungal diseases Saprolegniasis",
    4: "Healthy Fish",
    5: "Parasitic diseases",
    6: "Viral diseases White tail disease"
}

# Strong disease-specific symptoms
core_symptoms = {
    0: [
        "red patches on body",
        "bloody red spots",
        "red sores near fins",
        "bleeding marks on skin",
        "red ulcers on body",
        "inflamed red areas",
        "red streaks around tail",
        "small blood marks on scales",
        "skin looks reddish and damaged",
        "red wounds on belly"
    ],

    1: [
        "swollen belly",
        "body ulcers",
        "dropsy like swelling",
        "scales sticking out",
        "popeye",
        "bloated abdomen",
        "fluid filled belly",
        "deep skin ulcers",
        "red wounds with swelling",
        "fish looks bloated and weak"
    ],

    2: [
        "gasping at water surface",
        "pale gills",
        "swollen gills",
        "rapid gill movement",
        "mucus on gills",
        "breathing difficulty",
        "brown gill color",
        "fish opens mouth repeatedly",
        "stays near filter outlet",
        "gills look damaged"
    ],

    3: [
        "white cotton like growth",
        "fuzzy white patches",
        "fungus on body",
        "cotton wool growth",
        "white mold on fins",
        "fluffy fungus near wound",
        "grey white fungus",
        "fungal growth on skin",
        "white hairy patch",
        "mold like growth on tail"
    ],

    4: [
        "fish swimming normally",
        "active and eating well",
        "clear eyes",
        "normal breathing",
        "healthy fins",
        "bright body color",
        "smooth scales",
        "no visible wounds",
        "normal appetite",
        "fish looks healthy"
    ],

    5: [
        "white spots on body",
        "scratching against rocks",
        "rubbing body on tank wall",
        "parasites attached to skin",
        "tiny salt like dots",
        "clamped fins",
        "flashing behavior",
        "visible lice on fish",
        "skin irritation",
        "fish keeps scratching"
    ],

    6: [
        "tail turning white",
        "white tail tip",
        "tail tissue damaged",
        "milky white tail",
        "white discoloration on tail",
        "tail fin becoming pale",
        "tail rot like white area",
        "weak swimming with white tail",
        "white patch starting from tail",
        "tail end looks dead"
    ]
}

# Common symptoms that appear in many diseases
common_symptoms = [
    "not eating properly",
    "loss of appetite",
    "slow swimming",
    "staying at bottom",
    "hiding in corner",
    "weak movement",
    "less active than usual",
    "fish looks stressed",
    "abnormal swimming",
    "sometimes floating near surface",
    "breathing faster than normal",
    "color looks dull",
    "fins are clamped",
    "fish looks tired",
    "does not react much"
]

# Aquarium/water-quality noise
environment_noise = [
    "water is cloudy",
    "tank was cleaned yesterday",
    "ammonia may be high",
    "water temperature changed",
    "after water change",
    "new fish added recently",
    "filter was off for few hours",
    "oxygen level may be low",
    "fish was moved to new tank",
    "water smells bad",
    "tank has algae",
    "feed was changed recently",
    "water is slightly dirty",
    "pH may be unstable"
]

# Human style phrases
user_style_prefixes = [
    "",
    "my fish has",
    "i noticed",
    "fish showing",
    "problem is",
    "today i saw",
    "since yesterday",
    "for 2 days",
    "one fish has",
    "in my aquarium"
]

user_style_suffixes = [
    "",
    "what disease is this",
    "please help",
    "is it serious",
    "not sure what happened",
    "started recently",
    "getting worse",
    "only one fish affected",
    "other fish look normal",
    "it happens mostly at night"
]

# Realistic misspellings / typing mistakes
typo_map = {
    "disease": ["diesase", "desease", "disese"],
    "bacterial": ["bactrial", "bacteral"],
    "fungus": ["fungas", "fungis"],
    "white": ["wite", "whit"],
    "gills": ["gils", "gill"],
    "swollen": ["swolen", "swoleen"],
    "breathing": ["brething", "breathin"],
    "appetite": ["apetite", "appitite"],
    "scratching": ["scraching", "scratchng"],
    "parasites": ["parasite", "paracites"],
    "cotton": ["coton", "cottn"],
    "ulcers": ["ulcer", "ulcerss"],
    "bloody": ["blody", "blooddy"],
    "tail": ["teil", "taill"],
    "healthy": ["helthy", "healty"],
    "active": ["activ", "aktive"],
    "surface": ["surfac", "serface"]
}

# Words that may be randomly dropped
drop_words = [
    "the", "a", "is", "are", "has", "have", "and", "with", "on", "near", "like"
]


def introduce_typos(text, probability=0.12):
    words = text.split()
    new_words = []

    for word in words:
        clean_word = re.sub(r"[^a-zA-Z]", "", word.lower())

        if clean_word in typo_map and random.random() < probability:
            replacement = random.choice(typo_map[clean_word])
            new_words.append(replacement)
        else:
            new_words.append(word)

    return " ".join(new_words)


def randomly_drop_words(text, probability=0.08):
    words = text.split()
    kept_words = []

    for word in words:
        clean_word = re.sub(r"[^a-zA-Z]", "", word.lower())

        if clean_word in drop_words and random.random() < probability:
            continue

        kept_words.append(word)

    return " ".join(kept_words)


def random_punctuation_and_case(text):
    # Random punctuation
    if random.random() < 0.25:
        text = text.replace(" and ", ", ")

    if random.random() < 0.15:
        text += "..."

    if random.random() < 0.10:
        text += "?"

    # Random casing
    r = random.random()
    if r < 0.08:
        text = text.upper()
    elif r < 0.18:
        text = text.capitalize()

    return text


def shuffle_phrases(phrases):
    phrases = [p for p in phrases if p.strip()]
    random.shuffle(phrases)
    return " ".join(phrases)


def create_sample(label):
    disease_core = core_symptoms[label]

    phrases = []

    # Add prefix like real user input
    prefix = random.choice(user_style_prefixes)
    if prefix:
        phrases.append(prefix)

    # Healthy class should not always be perfectly clean
    if label == 4:
        phrases.append(random.choice(disease_core))

        # Add harmless noise
        if random.random() < 0.45:
            phrases.append(random.choice([
                "sometimes stays still",
                "eats a little slow but normal",
                "no red spots",
                "no white spots",
                "no fungus",
                "no swelling",
                "no breathing issue",
                "only resting after feeding",
                "swims fine most of the time"
            ]))

        if random.random() < 0.25:
            phrases.append(random.choice(environment_noise))

    else:
        # Add 1 or 2 strong symptoms
        phrases.append(random.choice(disease_core))

        if random.random() < 0.55:
            phrases.append(random.choice(disease_core))

        # Add common symptoms to make classes overlap
        if random.random() < 0.75:
            phrases.append(random.choice(common_symptoms))

        if random.random() < 0.45:
            phrases.append(random.choice(common_symptoms))

        # Add aquarium condition noise
        if random.random() < 0.45:
            phrases.append(random.choice(environment_noise))

        # Add mild symptom from another disease to create ambiguity
        if random.random() < 0.30:
            other_labels = [x for x in disease_labels.keys() if x != label and x != 4]
            other_label = random.choice(other_labels)
            phrases.append(random.choice(core_symptoms[other_label]))

    # Add suffix like real user query
    suffix = random.choice(user_style_suffixes)
    if suffix:
        phrases.append(suffix)

    # Some users write very short input
    if random.random() < 0.18:
        if label == 4:
            phrases = [random.choice(core_symptoms[label])]
        else:
            phrases = [
                random.choice(core_symptoms[label]),
                random.choice(common_symptoms)
            ]

    # Some users write only vague input
    if random.random() < 0.08 and label != 4:
        phrases = [
            random.choice(common_symptoms),
            random.choice(environment_noise)
        ]

    # Random phrase order
    if random.random() < 0.35:
        text = shuffle_phrases(phrases)
    else:
        text = " ".join(phrases)

    # Add spelling mistakes
    text = introduce_typos(text, probability=random.uniform(0.05, 0.18))

    # Drop small words
    text = randomly_drop_words(text, probability=random.uniform(0.03, 0.12))

    # Add punctuation/case variation
    text = random_punctuation_and_case(text)

    # Clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


rows = []

for label, disease_name in disease_labels.items():
    for _ in range(SAMPLES_PER_CLASS):
        symptom_text = create_sample(label)

        rows.append({
            "symptoms": symptom_text,
            "label": label,
            "disease": disease_name
        })

df = pd.DataFrame(rows)

# Shuffle final dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("Noisy realistic dataset generated successfully!")
print("File:", OUTPUT_FILE)
print("Shape:", df.shape)
print()
print(df.head(10))
print()
print("Class distribution:")
print(df["disease"].value_counts())