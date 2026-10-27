# Custom Feature Configuration

By default, `FeatureConfig.all()` enables all available linguistic feature extractors. You can fine-tune feature selection for specific stylometric experiments using `FeatureConfig`.

---

## Configuring Features

`FeatureConfig` provides fine-grained toggles for each stylistic category:

```python
from gram2vec import FeatureConfig, Gram2Vec

# Enable only POS tags and punctuation features
config = FeatureConfig(
    pos_tags=True,
    pos_bigrams=False,
    dep_labels=False,
    morph_tags=False,
    punctuation=True,
    emojis=False,
    syntax=False,
)

g2v = Gram2Vec(model="en_core_web_lg", config=config)
df = g2v.from_documents(["Hello, world! What a wonderful day."])
```

---

## Presets

`FeatureConfig` includes convenient presets:

```python
# Enable everything
config_all = FeatureConfig.all()

# Disable everything (useful as a baseline to selectively enable features)
config_none = FeatureConfig.none()
config_custom = config_none.enable("pos_tags", "punctuation")
```

---

## Custom Syntax Patterns with `SyntaxRegexMatcher`

The [`SyntaxRegexMatcher`](../reference/api.md#gram2vec.matcher.SyntaxRegexMatcher) matches linearized dependency trees against custom regular expressions:

```python
from gram2vec.matcher import SyntaxRegexMatcher

matcher = SyntaxRegexMatcher()
# Add custom syntactic patterns (e.g., passive voice, cleft sentences)
```
