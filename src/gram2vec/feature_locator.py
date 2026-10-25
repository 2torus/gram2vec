from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List

import demoji
from spacy.language import Language
from spacy.tokens import Doc

from .matcher import SyntaxRegexMatcher
from .vocab import Vocab


@dataclass
class FeatureSpan:
    """Represents a text span where a specific feature occurs."""

    feature_type: str
    feature_name: str
    text: str
    start_char: int
    end_char: int

    def __repr__(self) -> str:
        return f"{self.feature_type}:{self.feature_name} at char [{self.start_char}:{self.end_char}]: '{self.text}'"


class FeatureLocator:
    """Locates specific Gram2Vec features within text spans.
    Stateless and explicitly initialized with its dependencies.
    """

    def __init__(self, nlp: Language, vocab: Vocab, matcher: SyntaxRegexMatcher):
        self.nlp = nlp
        self.vocab = vocab
        self.matcher = matcher

        self.feature_handlers: Dict[str, Callable[[Doc, str, str], List[FeatureSpan]]] = {
            "pos_unigrams": self._locate_pos_unigrams,
            "pos_bigrams": self._locate_pos_bigrams,
            "func_words": self._locate_func_words,
            "punctuation": self._locate_punctuation,
            "letters": self._locate_letters,
            "dep_labels": self._locate_dep_labels,
            "morph_tags": self._locate_morph_tags,
            "sentences": self._locate_sentences,
            "emojis": self._locate_emojis,
        }

    def locate_feature(self, text: str, feature_name: str) -> List[FeatureSpan]:
        """Locates all occurrences of a specific feature in the text.

        Args:
            text: The input text to search in.
            feature_name: Feature in format 'feature_type:feature_value'
                          (e.g., 'pos_unigrams:NOUN', 'func_words:the').

        Returns:
            A list of FeatureSpan objects representing each occurrence.

        """
        if ":" not in feature_name:
            raise ValueError(f"Feature name must be in format 'feature_type:feature_value', got: '{feature_name}'")

        feature_type, feature_value = feature_name.split(":", 1)

        if feature_type not in self.feature_handlers:
            raise ValueError(
                f"Unknown feature type: '{feature_type}'. "
                f"Valid types are: {sorted(self.feature_handlers.keys())}"
            )

        doc = self.nlp(text)
        return self.feature_handlers[feature_type](doc, feature_value, text)

    def _locate_pos_unigrams(self, doc: Doc, feature_value: str, raw_text: str) -> List[FeatureSpan]:
        spans: List[FeatureSpan] = []
        for token in doc:
            if token.pos_ == feature_value:
                spans.append(
                    FeatureSpan(
                        feature_type="pos_unigrams",
                        feature_name=feature_value,
                        text=token.text,
                        start_char=token.idx,
                        end_char=token.idx + len(token.text),
                    )
                )
        return spans

    def _locate_pos_bigrams(self, doc: Doc, feature_value: str, raw_text: str) -> List[FeatureSpan]:
        spans: List[FeatureSpan] = []
        parts = feature_value.split(" ")
        if len(parts) != 2:
            return spans

        pos1, pos2 = parts

        for sent in doc.sents:
            sent_tokens = [token for token in sent]
            if not sent_tokens:
                continue

            # Check BOS bigram: BOS <pos2>
            if pos1 == "BOS" and sent_tokens[0].pos_ == pos2:
                spans.append(
                    FeatureSpan(
                        feature_type="pos_bigrams",
                        feature_name=feature_value,
                        text=sent_tokens[0].text,
                        start_char=sent_tokens[0].idx,
                        end_char=sent_tokens[0].idx + len(sent_tokens[0].text),
                    )
                )

            # Check EOS bigram: <pos1> EOS
            if pos2 == "EOS" and sent_tokens[-1].pos_ == pos1:
                spans.append(
                    FeatureSpan(
                        feature_type="pos_bigrams",
                        feature_name=feature_value,
                        text=sent_tokens[-1].text,
                        start_char=sent_tokens[-1].idx,
                        end_char=sent_tokens[-1].idx + len(sent_tokens[-1].text),
                    )
                )

            # Regular consecutive tokens
            for i in range(len(sent_tokens) - 1):
                t1 = sent_tokens[i]
                t2 = sent_tokens[i + 1]
                if t1.pos_ == pos1 and t2.pos_ == pos2:
                    start_char = t1.idx
                    end_char = t2.idx + len(t2.text)
                    spans.append(
                        FeatureSpan(
                            feature_type="pos_bigrams",
                            feature_name=feature_value,
                            text=raw_text[start_char:end_char],
                            start_char=start_char,
                            end_char=end_char,
                        )
                    )

        return spans

    def _locate_func_words(self, doc: Doc, feature_value: str, raw_text: str) -> List[FeatureSpan]:
        spans: List[FeatureSpan] = []
        func_words_vocab = set(self.vocab.get("func_words"))
        target = feature_value.lower()

        for token in doc:
            token_lower = token.text.lower()
            if token_lower == target and (token_lower in func_words_vocab or token.text in func_words_vocab):
                spans.append(
                    FeatureSpan(
                        feature_type="func_words",
                        feature_name=feature_value,
                        text=token.text,
                        start_char=token.idx,
                        end_char=token.idx + len(token.text),
                    )
                )
        return spans

    def _locate_punctuation(self, doc: Doc, feature_value: str, raw_text: str) -> List[FeatureSpan]:
        spans: List[FeatureSpan] = []
        punc_vocab = set(self.vocab.get("punctuation"))

        if feature_value not in punc_vocab:
            return spans

        for token in doc:
            for char_idx, char in enumerate(token.text):
                if char == feature_value:
                    abs_char_idx = token.idx + char_idx
                    spans.append(
                        FeatureSpan(
                            feature_type="punctuation",
                            feature_name=feature_value,
                            text=char,
                            start_char=abs_char_idx,
                            end_char=abs_char_idx + 1,
                        )
                    )
        return spans

    def _locate_letters(self, doc: Doc, feature_value: str, raw_text: str) -> List[FeatureSpan]:
        spans: List[FeatureSpan] = []
        letters_vocab = set(self.vocab.get("letters"))

        if feature_value not in letters_vocab:
            return spans

        for token in doc:
            for char_idx, char in enumerate(token.text):
                if char == feature_value:
                    abs_char_idx = token.idx + char_idx
                    spans.append(
                        FeatureSpan(
                            feature_type="letters",
                            feature_name=feature_value,
                            text=char,
                            start_char=abs_char_idx,
                            end_char=abs_char_idx + 1,
                        )
                    )
        return spans

    def _locate_dep_labels(self, doc: Doc, feature_value: str, raw_text: str) -> List[FeatureSpan]:
        spans: List[FeatureSpan] = []
        for token in doc:
            if token.dep_ == feature_value:
                spans.append(
                    FeatureSpan(
                        feature_type="dep_labels",
                        feature_name=feature_value,
                        text=token.text,
                        start_char=token.idx,
                        end_char=token.idx + len(token.text),
                    )
                )
        return spans

    def _locate_morph_tags(self, doc: Doc, feature_value: str, raw_text: str) -> List[FeatureSpan]:
        spans: List[FeatureSpan] = []
        for token in doc:
            if feature_value in token.morph:
                spans.append(
                    FeatureSpan(
                        feature_type="morph_tags",
                        feature_name=feature_value,
                        text=token.text,
                        start_char=token.idx,
                        end_char=token.idx + len(token.text),
                    )
                )
        return spans

    def _locate_sentences(self, doc: Doc, feature_value: str, raw_text: str) -> List[FeatureSpan]:
        spans: List[FeatureSpan] = []
        for sent in doc.sents:
            matches = self.matcher.match_document(sent.as_doc())
            for match in matches:
                if match.pattern_name == feature_value:
                    spans.append(
                        FeatureSpan(
                            feature_type="sentences",
                            feature_name=feature_value,
                            text=sent.text,
                            start_char=sent.start_char,
                            end_char=sent.end_char,
                        )
                    )
        return spans

    def _locate_emojis(self, doc: Doc, feature_value: str, raw_text: str) -> List[FeatureSpan]:
        spans: List[FeatureSpan] = []
        emojis_vocab = set(self.vocab.get("emojis"))

        if feature_value == "OOV_emoji":
            extracted_emojis = demoji.findall_list(raw_text, desc=False)
            for emoji in extracted_emojis:
                if emoji not in emojis_vocab:
                    for m in re.finditer(re.escape(emoji), raw_text):
                        spans.append(
                            FeatureSpan(
                                feature_type="emojis",
                                feature_name="OOV_emoji",
                                text=emoji,
                                start_char=m.start(),
                                end_char=m.end(),
                            )
                        )
        else:
            for match in re.finditer(re.escape(feature_value), raw_text):
                spans.append(
                    FeatureSpan(
                        feature_type="emojis",
                        feature_name=feature_value,
                        text=feature_value,
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )

        return spans
