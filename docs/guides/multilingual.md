# Multilingual Support

Gram2Vec supports multiple languages through language-specific spaCy models and vocabularies.

---

## Supported Languages

| Language | Model | Vocabularies & Morph Tags |
| :--- | :--- | :--- |
| **English** | `en_core_web_lg` | Full UD POS, morph tags, syntax patterns, emojis, punctuation |
| **Russian** | `ru_core_news_lg` | Full Russian UD POS, case/gender/aspect morphology, Russian vocabularies |
| **Chinese** | `zh_core_web_sm` | Segmentation, POS tags, and dependency parsing |

---

## Russian Vectorization Example

```python
from gram2vec import Gram2Vec, FeatureConfig

# Initialize Gram2Vec with Russian model
g2v_ru = Gram2Vec(model="ru_core_news_lg", config=FeatureConfig.all())

russian_texts = [
    "Быть или не быть, вот в чём вопрос.",
    "Я помню чудное мгновенье: передо мной явилась ты.",
]

features_ru = g2v_ru.from_documents(russian_texts)
print(features_ru.head())
```

---

## Verbalizing Features in Russian

The [`Verbalizer`](../reference/api.md#gram2vec.verbalizer.Verbalizer) translates cryptic feature names into human-readable descriptions in either English or Russian:

```python
from gram2vec import Verbalizer

# Translate Russian feature names into human-readable Russian descriptions
verbalizer = Verbalizer(language="ru")
desc = verbalizer.verbalize("pos_VERB")
print(desc)  # Descriptive Russian name for verb feature
```
