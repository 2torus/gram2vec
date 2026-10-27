# Stylistic Feature Catalog

Gram2Vec extracts tangible stylistic features organized into distinct linguistic families. Every feature is computed as a **relative frequency normalized by the total number of tokens** (or sentences, where appropriate) in the document.

---

## 1. Part-of-Speech (POS) Tags

Gram2Vec uses the 17 Universal Dependencies (UD) POS tags:

| Tag | Linguistic Name | Description & Example |
| :--- | :--- | :--- |
| `pos_ADJ` | Adjective | Modifies nouns (`big`, `red`, `quick`) |
| `pos_ADP` | Adposition | Preposition or postposition (`in`, `to`, `during`) |
| `pos_ADV` | Adverb | Modifies verbs, adjectives, or other adverbs (`very`, `well`, `quickly`) |
| `pos_AUX` | Auxiliary | Auxiliary verb (`is`, `has`, `will`, `should`) |
| `pos_CCONJ` | Coordinating Conjunction | Connects equal grammatical elements (`and`, `or`, `but`) |
| `pos_DET` | Determiner | Expresses reference of a noun phrase (`the`, `a`, `this`) |
| `pos_INTJ` | Interjection | Expresses emotional state (`ouch`, `hello`, `yes`) |
| `pos_NOUN` | Noun | Common noun (`dog`, `computer`, `time`) |
| `pos_NUM` | Numeral | Number or quantity (`one`, `42`, `third`) |
| `pos_PART` | Particle | Functional word that doesn't fit standard categories (`not`, `'s`) |
| `pos_PRON` | Pronoun | Substitutes for a noun phrase (`she`, `they`, `someone`) |
| `pos_PROPN` | Proper Noun | Specific name of a person, place, or organization (`Paris`, `Google`) |
| `pos_PUNCT` | Punctuation | Punctuation mark (`,`, `.`, `?`, `!`) |
| `pos_SCONJ` | Subordinating Conjunction | Introduces a subordinate clause (`because`, `although`, `if`) |
| `pos_SYM` | Symbol | Mathematical or typographic symbol (`$`, `%`, `+`) |
| `pos_VERB` | Verb | Main action, occurrence, or state of being (`run`, `think`, `exist`) |
| `pos_X` | Other | Foreign words, abbreviations, or unclassified tokens |

---

## 2. POS Bigrams

Consecutive pairs of POS tags (e.g., `pos_NOUN_VERB`, `pos_DET_NOUN`). With 18 tags (including spaces), there are $18^2 = 324$ possible combinations capturing local syntactic transitions and cadence.

---

## 3. Morphological Features

Morphological features describe grammatical inflection, gender, number, tense, and mood:

| Feature | Category | Meaning |
| :--- | :--- | :--- |
| `morph_Aspect=Perf` | Aspect | Perfect aspect |
| `morph_Aspect=Prog` | Aspect | Progressive / continuous aspect |
| `morph_Case=Nom` | Case | Nominative case |
| `morph_Case=Acc` | Case | Accusative case |
| `morph_Definite=Def` | Definiteness | Definite article (`the`) |
| `morph_Definite=Ind` | Definiteness | Indefinite article (`a`, `an`) |
| `morph_Degree=Cmp` | Degree | Comparative degree (`faster`, `better`) |
| `morph_Degree=Sup` | Degree | Superlative degree (`fastest`, `best`) |
| `morph_Gender=Fem` | Gender | Feminine gender |
| `morph_Gender=Masc` | Gender | Masculine gender |
| `morph_Mood=Ind` | Mood | Indicative mood |
| `morph_Number=Sing` | Number | Singular number |
| `morph_Number=Plur` | Number | Plural number |
| `morph_Person=1` | Person | 1st person (`I`, `we`) |
| `morph_Person=2` | Person | 2nd person (`you`) |
| `morph_Person=3` | Person | 3rd person (`he`, `she`, `they`) |
| `morph_Tense=Past` | Tense | Past tense |
| `morph_Tense=Pres` | Tense | Present tense |
| `morph_VerbForm=Fin` | Verb Form | Finite verb |
| `morph_VerbForm=Inf` | Verb Form | Infinitive verb |

---

## 4. Syntactic Dependency Labels

Universal Dependencies syntactic relations representing head-dependent connections:

| Label | Role |
| :--- | :--- |
| `dep_nsubj` | Nominal subject |
| `dep_obj` / `dep_dobj` | Direct object |
| `dep_iobj` | Indirect object |
| `dep_advcl` | Adverbial clause modifier |
| `dep_amod` | Adjectival modifier |
| `dep_compound` | Compound noun or multi-word expression |
| `dep_root` | Root predicate of the sentence |

---

## 5. Punctuation & Emojis

* **Punctuation Marks**: Normalized relative counts of commas, periods, exclamation points, question marks, semicolons, colons, quotes, dashes, and parentheses.
* **Emojis**: Extracted via `demoji` and normalized by token count.
