"""Syntax regex matcher for syntactic patterns on dependency linearized trees."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from spacy.tokens import Doc, Span


@dataclass
class Match:
    """Represents a regex match on a linearized sentence."""

    pattern_name: str
    matched: str
    sentence: str

    def __repr__(self) -> str:
        return f'{self.pattern_name} : {self.sentence}'


DEFAULT_ENGLISH_PATTERNS: Dict[str, str] = {
    'it-cleft': r'\([^-]*-be-[^-]*-[^-]*.*\([iI]t-it-PRP-nsubj\).*\([^-]*-[^-]*-NN[^-]*-attr.*\([^-]*-[^-]*-VB[^-]*-(relcl|advcl)',
    'pseudo-cleft': r'\([^-]*-be-[^-]*-[^-]*.*\([^-]*-[^-]*-(WP|WRB)-(dobj|advmod)',
    'all-cleft': r'(\([^-]*-be-[^-]*-[^-]*.*\([^-]*-all-(P)?DT)|(\([^-]*-all-(P)?DT-[^-]*.*\([^-]*-be-[^-]*)',
    'there-cleft': r'\([^-]*-be-[^-]*-[^-]*.*\([^-]*-there-EX-expl.*\([^-]*-[^-]*-[^-]*-attr.*\([^-]*-[^-]*-[^-]*-(relcl|acl)',
    'if-because-cleft': r'\([^-]*-be-[^-]*-[^-]*.*\([^-]*-[^-]*-[^-]*-advcl\([^-*]*-if-IN-mark',
    'passive': r'\([^-]*-[^-]*-(NN[^-]*|PRP|WDT)-nsubjpass.*\([^-]*-be-[^-]*-auxpass',
    'subj-relcl': r'\([^-]*-[^-]*-[^-]*-relcl.*\([^-]*-[^-]*-(WP|WDT)-nsubj',
    'obj-relcl': r'\([^-]*-[^-]*-NN[^-]*-(nsubj|attr).*\([^-]*-[^-]*-[^-]*-(relcl|ccomp).*\([^-]*-[^-]*-(WP|WDT|IN)-(pobj|dobj)',
    'tag-question': r'\([^-]*-(do|be|could|can|have)-[^-]*-ROOT.*\(\?-\?-\.-punct',
    'coordinate-clause': r'\([^-]*-[^-]*-CC-cc\).*\([^-]*-[^-]*-(VB[^-]*|JJ)-conj.*\([^-]*-[^-]*-[^-]*-nsubj',
}

DEFAULT_RUSSIAN_PATTERNS: Dict[str, str] = {
    'passive_rus': r'\([^-]*-[^-]*-VERB-ROOT.*?Voice=Pass.*?\)+$',
    'parataxis_rus': r'\([^-]*-[^-]*-[^-]*-parataxis.*\)',
    'participle_rus': r'\([^-]*-[^-]*-(?:VERB|ADJ)-ROOT.*?(?:ADJ|VERB)-(?:amod|acl)(?!:relcl).*?(?:VerbForm=Part|[а-я]+(?:ющ|ащ|ящ|вш|ем|им|нн|т)[а-я]+).*?\)+$',
    'gerund_rus': r'\([^-]*-[^-]*-(?:VERB|PROPN)-ROOT.*?(?:[А-Яа-я]+(?<!ющ|[ая]щ|вш)[яв(сь|ся)]-[^-]*-(?:VERB|NOUN|ADJ|PROPN)-(?:advcl|amod).*?(?:VerbForm=Conv|[^|]*)).*?\)+$',
    'conj_rus': r'\([^-]*-[^-]*-(?!SCONJ)\w+-ROOT.*?-conj.*?\)+$',
    'nested_structure_rus': r'\([^-]*-[^-]*-(?:VERB|ADV|NOUN)-ROOT.*?(?:VERB-(?:xcomp|advcl|ccomp|acl:relcl|acl)|AUX-aux:pass|ADV-advcl).*?(?:SCONJ-(?:mark|fixed|obj)|ADV-(?:mark|advmod)|который-PRON|что-PRON|чтобы-SCONJ|хотя-SCONJ|такой-DET).*?\)+$',
    'one_word_sent_rus': r'\([^-]*-[^-]*-(NOUN|VERB|PROPN)-ROOT-[^()]*\([^-]*-[^-]*-PUNCT-punct-\)\)',
    'diminutive_rus': r'.*\([а-яё]*(?:ик|[её]к|[её]нок|очк|ечк|ышк|оньк|еньк)[а-яё]*-[а-яё]*-NOUN-.*?\).*',
    'multiple_punct_rus': r'\([^-]*-[^-]*-[^-]*-ROOT.*?(?:(?:\(([!?])\1*-\1\1*-PUNCT-punct-\)){2,}|\(\.{2,}-\.{2,}-PUNCT-punct-\))\)+$',
    'additional_info_rus': r'\([^-]*-[^-]*-[^-]*-ROOT.*?(?:\([^-]*\([^)]+\)[^-]*-[^-]*-PUNCT-punct-[^)]*\)|(?:\([^-]*-[^-]*-PUNCT-punct-[^)]*\))){2,}\)+$',
}


class SyntaxRegexMatcher:
    """Encapsulates sentence regex patterns and methods to apply them to target documents."""

    def __init__(
        self,
        language: str = 'en',
        patterns: Optional[Dict[str, str]] = None,
    ):
        self.language = language

        if patterns is not None:
            self.patterns = dict(patterns)
        elif language == 'en':
            self.patterns = dict(DEFAULT_ENGLISH_PATTERNS)
        elif language == 'ru':
            self.patterns = dict(DEFAULT_RUSSIAN_PATTERNS)
        else:
            self.patterns = {}

    def print_patterns(self) -> None:
        """Prints all registered patterns."""
        for pattern_name, pattern in self.patterns.items():
            print(f'{pattern_name} : {pattern}\n')

    def _find_treegex_matches(self, doc: Doc) -> Tuple[Match, ...]:
        """Iterates through a document's sentences, applying every regex to each sentence."""
        matches: List[Match] = []
        for sent in doc.sents:
            tree_string = self.linearize_tree(sent)
            for name, pattern in self.patterns.items():
                match = re.search(pattern, tree_string)
                if match:
                    matches.append(Match(name, match.group(), sent.text))
        return tuple(matches)

    def add_patterns(self, patterns: Dict[str, str]) -> None:
        """Updates patterns with a user supplied dictionary of {pattern_name: regex} pairs."""
        self.patterns.update(patterns)

    def remove_patterns(self, to_remove: Iterable[str]) -> None:
        """Removes the given pattern names from registered patterns."""
        for pattern_name in to_remove:
            if pattern_name not in self.patterns:
                raise KeyError(f"Pattern '{pattern_name}' not in registered patterns.")
            del self.patterns[pattern_name]

    def match_document(self, document: Doc) -> Tuple[Match, ...]:
        """Applies all registered patterns to a single spaCy document."""
        return self._find_treegex_matches(document)

    def match_documents(self, documents: Iterable[Doc]) -> List[Tuple[Match, ...]]:
        """Applies all registered patterns to a collection of spaCy documents."""
        return [self._find_treegex_matches(doc) for doc in documents]

    def linearize_tree(self, sentence: Span) -> str:
        """Converts a spaCy dependency-parsed sentence into a linear tree string while preserving relations."""

        def get_nt_count(sent: Span) -> int:
            return sum(1 for token in sent if list(token.children))

        def ending_parenthesis(n: int) -> str:
            return ')' * n

        def parse_dependency_parse(sent: Span) -> str:
            stack = [sent.root]
            result = ''
            while stack:
                token = stack.pop()
                if self.language == 'en':
                    result += f'({token.text}-{token.lemma_}-{token.tag_}-{token.dep_}'
                else:
                    # Russian and general morphological representation
                    result += f'({token.text}-{token.lemma_}-{token.tag_}-{token.dep_}-{token.morph}'

                stack.extend(reversed(list(token.children)))

                if not list(token.children):
                    result += ')'
            return result

        parse = parse_dependency_parse(sentence)
        nt_count = get_nt_count(sentence)
        return f'{parse}{ending_parenthesis(nt_count)}'
