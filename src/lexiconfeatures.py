import re
import scipy.sparse as sp
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

RACIAL_SLURS = {
      "negro", "negros", "nigger", "niggers", "nigga", "niggas",
    "chink", "chinks", "spic", "spics", "wetback", "wetbacks",
    "gook", "gooks", "kike", "kikes", "towelhead", "raghead",
    "cracker", "honky", "coon", "coons", "sandnigger",
}

IDENTITY_TARGET_TERMS = {
      "immigrants", "immigrant", "muslims", "muslim", "islam",
    "jews", "jewish", "jew", "christians", "women", "woman",
    "gays", "gay", "homosexuals", "homosexual", "transgender",
    "chinese", "mexicans", "mexican", "blacks", "whites",
    "refugees", "refugee", "foreigners", "foreigner",
    "hispanic", "latinos", "latina", "latino", "arabs", "arab",
    "indians", "indian",
}

GENERIC_INSULTS = {
    "stupid", "idiot", "moron", "dumb", "fool", "loser",
    "jerk", "asshole", "bastard", "bitch", "dick",
    "retard", "imbecile", "clown", "pathetic",
}


EXCLUSION_PHRASES = {
    r"should (not|never) be allowed",
    r"don'?t belong",
    r"go back (to|where)",
    r"get out of",
    r"shouldn'?t exist",
    r"should be (banned|removed|deported|eliminated)",
    r"(hate|despise|loathe) all",
    r"are (inferior|subhuman|animals|vermin|parasites)",
    r"(not|never) welcome here",
}

class LexiconFeatures(BaseEstimator, TransformerMixin):
    """Explicit hate-speech signals TF-IDF alone cannot reliably learn"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            text = str(text).lower()
            tokens = set(text.split())
            has_slur = float(bool(tokens & RACIAL_SLURS))
            has_target = float(bool(tokens & IDENTITY_TARGET_TERMS))
            has_exclusion = float(
                any(re.search(pattern, text) for pattern in EXCLUSION_PHRASES)
            )

            # Strongest single signal: slur + identity group in one sentence.
            slur_and_target = float(has_slur and has_target)
            # Second strongest signal: identity group + exclusion phrase.
            excl_and_target = float(has_exclusion and has_target)


            generic_no_target = float(
                bool(tokens & GENERIC_INSULTS) and not has_target and not has_slur
            )

            is_positive = float(
                bool(re.search(
                    r"\b(good|great|decent|welcome|wonderful|nice|kind|love)\b",
                    text,
                )) and not has_slur and not has_exclusion
            )

            rows.append([
                has_slur,
                has_target,
                has_exclusion,
                slur_and_target,
                excl_and_target,
                generic_no_target,
                is_positive,
            ])

        return sp.csr_matrix(np.array(rows, dtype=float))
