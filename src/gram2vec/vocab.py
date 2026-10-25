from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources  # type: ignore


STANDARD_VOCAB_NAMES = (
    "pos_unigrams",
    "pos_bigrams",
    "dep_labels",
    "emojis",
    "func_words",
    "morph_tags",
    "punctuation",
    "letters",
)


class Vocab:
    """Encapsulates feature vocabularies without relying on global state or environment variables.
    """

    def __init__(
        self,
        language: str = "en",
        vocab_dir: Optional[Union[str, Path]] = None,
        load_defaults: bool = True,
        matcher: Optional[Any] = None,
    ):
        self.language = language
        self._vocabs: dict[str, List[str]] = {}
        self.vocab_dir = self._resolve_vocab_dir(vocab_dir)

        if load_defaults:
            self.load_default_vocabs(matcher=matcher)

    def _resolve_vocab_dir(self, vocab_dir: Optional[Union[str, Path]]) -> Path:
        """Resolves the directory containing vocabulary files."""
        if vocab_dir is not None:
            path = Path(vocab_dir)
            if not path.is_dir():
                raise NotADirectoryError(f"Provided vocab directory does not exist: {path}")
            return path

        # Default to built-in vocab package directory
        try:
            return Path(str(pkg_resources.files("gram2vec").joinpath("vocab")))
        except Exception:
            # Fallback for development / non-packaged environments
            return Path(__file__).resolve().parent / "vocab"

    def _load_from_txt(self, path: Union[str, Path]) -> List[str]:
        """Loads a .txt file delimited by newlines."""
        with open(path, "r", encoding="utf-8") as fin:
            return [line.rstrip("\r\n") for line in fin]

    def get(self, vocab_name: str) -> List[str]:
        """Fetches the desired vocabulary items."""
        if vocab_name in self._vocabs:
            return self._vocabs[vocab_name]
        raise KeyError(f"Requested vocab '{vocab_name}' not found in registered vocabularies: {list(self._vocabs.keys())}")

    def has(self, vocab_name: str) -> bool:
        """Checks if a vocabulary name is registered."""
        return vocab_name in self._vocabs

    def add_from_path(
        self,
        vocab_name: str,
        language: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Adds vocabulary from a newline delimited .txt file.
        If file_path is explicitly given, loads that file directly.
        Otherwise, attempts to find '<language>_<vocab_name>.txt' or falls back to '<vocab_name>.txt'
        inside the configured vocab directory.
        """
        if file_path is not None:
            path = Path(file_path)
            if not path.is_file():
                raise FileNotFoundError(f"Vocab file '{path}' does not exist")
            self._vocabs[vocab_name] = self._load_from_txt(path)
            return

        target_lang = language if language is not None else self.language

        # Try language-specific file first if not default 'en'
        if target_lang and target_lang != "en":
            lang_path = self.vocab_dir / f"{target_lang}_{vocab_name}.txt"
            if lang_path.is_file():
                self._vocabs[vocab_name] = self._load_from_txt(lang_path)
                return

        # Fallback to default file
        default_path = self.vocab_dir / f"{vocab_name}.txt"
        if not default_path.is_file():
            raise FileNotFoundError(
                f"Vocab file '{default_path}' does not exist for vocab '{vocab_name}'"
            )

        self._vocabs[vocab_name] = self._load_from_txt(default_path)

    def add_items(self, vocab_name: str, items: Iterable[str]) -> None:
        """Adds vocabulary items directly given a vocab name and iterable of strings."""
        self._vocabs[vocab_name] = list(items)

    def load_default_vocabs(self, matcher: Optional[Any] = None) -> None:
        """Loads all standard vocabularies for the configured language."""
        for name in STANDARD_VOCAB_NAMES:
            self.add_from_path(name, language=self.language)

        # Register sentence patterns if matcher is provided
        if matcher is not None and hasattr(matcher, "patterns"):
            self.add_items("sentences", tuple(matcher.patterns.keys()))
        else:
            # Lazy import to avoid circular dependency
            from .matcher import SyntaxRegexMatcher
            try:
                temp_matcher = SyntaxRegexMatcher(language=self.language)
                self.add_items("sentences", tuple(temp_matcher.patterns.keys()))
            except Exception:
                # If language is unknown and no patterns provided, default to empty
                self.add_items("sentences", ())
