"""
Test script demonstrating the explicit Gram2Vec API
"""

from multiprocessing import freeze_support
from gram2vec import FeatureConfig, Gram2Vec

config = FeatureConfig.all()

example_sentences = [
    "The quick brown 😂 fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The early bird catches the worm.",
    "A picture is worth a thousand words.",
    "When in Rome, do as the Romans do.",
]


def main():
    # Explicitly initialize Gram2Vec (supports package name, Path to local folder, or spacy Language)
    g2v = Gram2Vec(model="en_core_web_lg", config=config)

    # Extract features from example sentences
    print("Vectorizing example sentences...")
    features = g2v.from_documents(example_sentences)
    features.to_csv("features.csv", index=False)
    print(f"Extracted feature shape: {features.shape}")

    # Find occurrences of specific features in a text
    example_text = "The quick brown 😂 fox jumps over the lazy dog."
    print(f"\nAnalyzing text: \"{example_text}\"")

    # Find all nouns
    noun_spans = g2v.find_feature_spans(example_text, "pos_unigrams:NOUN")
    print("\nNouns found:")
    for span in noun_spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find all determiners
    det_spans = g2v.find_feature_spans(example_text, "pos_unigrams:DET")
    print("\nDeterminers found:")
    for span in det_spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find function words
    the_spans = g2v.find_feature_spans(example_text, "func_words:the")
    print("\nFunction word 'the' found:")
    for span in the_spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find emojis
    emoji_spans = g2v.find_feature_spans(example_text, "emojis:😂")
    print("\nEmojis found:")
    for span in emoji_spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find punctuation
    punct_spans = g2v.find_feature_spans(example_text, "punctuation:.")
    print("\nPunctuation '.' found:")
    for span in punct_spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find noun->verb sequences
    spans = g2v.find_feature_spans(example_text, "pos_bigrams:NOUN VERB")
    print("\nNoun->verb sequences found:")
    for span in spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find sentences starting with adjectives
    spans = g2v.find_feature_spans(example_text, "pos_bigrams:BOS ADJ")
    print("\nSentences starting with adjectives found:")
    for span in spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find sentences ending with punctuation
    spans = g2v.find_feature_spans(example_text, "pos_bigrams:PUNCT EOS")
    print("\nSentences ending with punctuation found:")
    for span in spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")


if __name__ == "__main__":
    freeze_support()
    main()
