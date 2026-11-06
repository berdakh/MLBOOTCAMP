"""Model artifacts that can be trusted after the notebook is gone.

A *bundle* is a directory::

    model_v3/
        model.joblib      the fitted sklearn Pipeline (preprocessing included)
        metadata.json     everything needed to judge and safely load it

The metadata answers the questions you will be asked about any model: what
data was it trained on (hash), which features in which order, which library
versions, what threshold, what validation metrics, and when.

Security note: ``joblib``/``pickle`` files execute code when loaded. Only load
bundles you created or obtained from a source you trust.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """A stable short hash of a DataFrame's contents (for 'trained on what?').

    Text columns are cast to plain ``object`` first: pandas 3 stores them as a
    ``str`` dtype that hashes differently from pandas 2's ``object``, and the
    same rows must give the same fingerprint on either version.
    """
    text = df.select_dtypes(include=["object", "string"]).columns
    df = df.astype({c: object for c in text})
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


@dataclass
class ModelMetadata:
    model_name: str
    version: str
    features: list[str]
    categorical: list[str]
    threshold: float
    label: str = "churned_30d"
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    training_data: dict = field(default_factory=dict)  # fingerprint, rows, period
    metrics: dict = field(default_factory=dict)  # validation metrics only!
    params: dict = field(default_factory=dict)
    libraries: dict = field(default_factory=lambda: {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
    })
    notes: str = ""


class ArtifactError(RuntimeError):
    """Raised when a bundle is missing, incomplete or incompatible."""


def save_bundle(model, metadata: ModelMetadata, directory: Path, overwrite: bool = False) -> Path:
    """Write ``model`` and ``metadata`` atomically-ish into ``directory``."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise ArtifactError(f"{directory} already exists; bundles are immutable - use a new version")
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    meta = asdict(metadata)
    meta["model_sha256"] = _sha256(directory / "model.joblib")
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return directory


def load_metadata(directory: Path) -> dict:
    path = Path(directory) / "metadata.json"
    if not path.exists():
        raise ArtifactError(f"no metadata.json in {directory}")
    return json.loads(path.read_text())


def load_bundle(directory: Path, strict_versions: bool = False):
    """Load (model, metadata), verifying the file hash and library versions.

    A sklearn major/minor mismatch is a warning by default and an error with
    ``strict_versions=True``: pickles are not guaranteed to load correctly
    across sklearn versions, and failures can be silent.
    """
    directory = Path(directory)
    meta = load_metadata(directory)
    model_path = directory / "model.joblib"
    if not model_path.exists():
        raise ArtifactError(f"no model.joblib in {directory}")
    if _sha256(model_path) != meta.get("model_sha256"):
        raise ArtifactError("model.joblib does not match the hash in metadata.json - corrupted or tampered")
    trained = meta["libraries"]["sklearn"].split(".")[:2]
    running = sklearn.__version__.split(".")[:2]
    if trained != running:
        msg = f"model trained with sklearn {meta['libraries']['sklearn']}, running {sklearn.__version__}"
        if strict_versions:
            raise ArtifactError(msg)
        import warnings
        warnings.warn(msg, stacklevel=2)
    return joblib.load(model_path), meta


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
