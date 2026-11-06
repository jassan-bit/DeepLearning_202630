"""Read the current documentary crypto extension without reading private originals."""
import hashlib
import json
from pathlib import Path
import re

STAGE_ID = "crypto_extension_20261105"
METADATA_SUFFIXES = {".json", ".csv", ".svg", ".bib", ".md"}
COUNT_KEYS = {
    "bibliographic_references", "crypto_full_text_studies", "BTC_full_text_studies",
    "studies_with_non_BTC_crypto", "prior_full_text_studies", "new_crypto_inclusions",
    "MLP_baseline_studies", "CNN_generic_baseline_studies",
    "CNN1D_explicit_baseline_studies", "LSTM_baseline_studies",
    "neural_comparator_extractable_studies",
}
LEGACY_BIB_KEYS = (
    "miura2019", "huang2024", "akbiyik2023", "guo2018", "barjasic2021",
    "katsiampa2017", "dyhrberg2016", "herremans2025", "cybenko1989", "goodfellow2016",
    "hochreiter1997", "cho2014", "lecun1998", "bai2018", "kingma2014", "engle1982",
    "bollerslev1986", "andersen2003", "corsi2009", "diebold1995", "harvey1997",
    "hyndman2006", "bergmeir2018", "holm1979", "kunsch1989", "politis1994",
    "haddaway2022", "dickey1979", "kwiatkowski1992", "ljung1978", "broock1996",
    "vaswani2017", "zeri2026", "gao2023", "wangnd", "alakhouch2026",
    "bucci2020realized", "reisenhofer2022harnet", "morenopino2024deepvol",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate_bibliography(root, stage=None):
    """Retain the historical catalog and allow a separately verified new cut.

    The 05/11 catalog is an immutable evidence cut, not a ceiling on later
    bibliography entries. New original readings must have a hashed BibTeX
    fragment; they never change that cut's corpus or family frequencies.
    """
    root = Path(root)
    keys = re.findall(r"@\w+\s*\{\s*([^,]+),", (root / "book/references.bib").read_text())
    previous = read(root / "references/literature_e1/journal_update_20261105/references_access_39.json")
    inherited = [row["key"] for row in previous["references"]]
    historical_count = stage["counts"]["bibliographic_references"] if stage else 39
    extra_keys = []
    review_dir = root / "references/literature_e1/primary_reviews_20261106"
    fragment = review_dir / "references.bib"
    if stage and fragment.is_file():
        manifest = read(review_dir / "manifest.json")
        expected_hash = manifest["files_sha256"].get("references.bib")
        if fragment.is_symlink() or sha(fragment) != expected_hash:
            raise ValueError("The new bibliography fragment must match its review manifest")
        extra_keys = re.findall(r"@\w+\s*\{\s*([^,]+),", fragment.read_text())
        if not extra_keys or len(extra_keys) != len(set(extra_keys)):
            raise ValueError("The new review fragment needs unique reference keys")
    expected_count = historical_count + len(extra_keys)
    if (len(inherited) != 39 or len(set(inherited)) != 39 or len(keys) != expected_count
            or len(set(keys)) != len(keys) or set(keys[:39]) != set(inherited)
            or tuple(keys[:39]) != LEGACY_BIB_KEYS):
        raise ValueError("Preserve the 39 inherited bibliography keys and their order")
    if stage:
        catalog = read(stage["directory"] / "references_access_40.json")
        catalog_keys = [row["key"] for row in catalog["references"]]
        if (len(catalog_keys) != historical_count
                or set(catalog_keys) != set(keys[:historical_count])):
            raise ValueError("The historical reference access catalog must match its bibliography prefix")
        if keys[39:historical_count] != ["djanga2025commonality"]:
            raise ValueError("Preserve the documented 05/11 crypto reference")
        if keys[historical_count:] != extra_keys:
            raise ValueError("Later bibliography entries must match the separately verified review fragment")
    return keys


def load_crypto_stage(root):
    """Return a verified metadata stage, or None when the new stage is absent."""
    root = Path(root)
    directory = root / "references/literature_e1" / STAGE_ID
    if not directory.exists():
        return None
    public = root / "book/_static/literature_e1" / STAGE_ID
    summary_path, manifest_path = directory / "summary.json", directory / "manifest.json"
    if (directory.is_symlink() or public.is_symlink() or manifest_path.is_symlink()
            or (public / "manifest.json").is_symlink()):
        raise ValueError("Expected regular documentary literature directories")
    summary, manifest = read(summary_path), read(manifest_path)
    if summary.get("schema_version") != 1 or summary.get("stage_id") != STAGE_ID:
        raise ValueError("Unexpected crypto literature stage schema or identifier")
    counts = summary["counts"]
    if not COUNT_KEYS.issubset(counts) or any(type(counts[key]) is not int or counts[key] < 0 for key in COUNT_KEYS):
        raise ValueError("The crypto stage must declare nonnegative integer corpus counts")
    total = counts["crypto_full_text_studies"]
    if (counts["BTC_full_text_studies"] > total or counts["studies_with_non_BTC_crypto"] > total
            or counts["prior_full_text_studies"] + counts["new_crypto_inclusions"] != total
            or len(summary["study_ids"]) != total or len(set(summary["study_ids"])) != total
            or len(summary["new_study_ids"]) != counts["new_crypto_inclusions"]
            or not set(summary["new_study_ids"]).issubset(summary["study_ids"])
            or any(counts[key] > total for key in COUNT_KEYS if key.endswith("baseline_studies"))
            or counts["neural_comparator_extractable_studies"] > total):
        raise ValueError("Inconsistent crypto corpus or comparator counts")
    if (summary.get("timing") != "post-benchmark documentary extension"
            or summary.get("experiment_results_modified") is not False
            or summary.get("new_model_fits") != 0 or summary.get("new_forecasts") != 0):
        raise ValueError("The crypto extension must retain its documentary, post-benchmark scope")
    erratum = summary["harnet_erratum_file"]
    hashes = manifest["files_sha256"]
    if not {"summary.json", "references_access_40.json", erratum}.issubset(hashes):
        raise ValueError("The crypto manifest must include its summary, catalog and HARNet erratum")
    source_hashes = {}
    for name, expected in hashes.items():
        relative = Path(name)
        path = directory / relative
        mirrored = public / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(directory.resolve())
                or relative.suffix.lower() not in METADATA_SUFFIXES
                or relative.name == "manifest.json"):
            raise ValueError("Unexpected non-metadata or unsafe crypto artifact: " + name)
        if (sha(path) != expected or sha(mirrored) != expected or mirrored.is_symlink()
                or not mirrored.resolve().is_relative_to(public.resolve())):
            raise ValueError("Changed crypto source or public mirror: " + name)
        source_hashes[path.relative_to(root).as_posix()] = expected
        source_hashes[mirrored.relative_to(root).as_posix()] = expected
    if sha(manifest_path) != sha(public / "manifest.json"):
        raise ValueError("The crypto source and public manifest must be byte-identical")
    for folder in (directory, public):
        actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
        if actual != set(hashes) | {"manifest.json"}:
            raise ValueError("Every crypto metadata artifact must appear in its manifest")
        source_hashes[(folder / "manifest.json").relative_to(root).as_posix()] = sha(folder / "manifest.json")
    stage = {"summary": summary, "counts": counts, "directory": directory,
             "public_directory": public, "manifest": manifest, "source_sha256": source_hashes}
    validate_bibliography(root, stage)
    source_hashes["book/references.bib"] = sha(root / "book/references.bib")
    return stage


def crypto_extension(stage):
    """Delivery-compatible current assessment, separate from the historical journal cut."""
    summary, counts = stage["summary"], stage["counts"]
    return {
        "stage_id": STAGE_ID, "updated_at_utc": summary["compiled_at_utc"], **counts,
        "timing": summary["timing"], "study_ids": summary["study_ids"],
        "new_study_ids": summary["new_study_ids"],
        "new_BTC_full_text_inclusions": counts["new_crypto_inclusions"],
        "new_methodological_full_texts": 1, "new_methodological_author_code_sources": 1,
        "new_publication_identity_only_sources": 1,
        "T1": "Bucci2020 journal original; standalone feedforward reference; S&P500 monthly",
        "T2": "HARNet2022 author Conv1D code only; scaler erratum corrects an unsupported leakage attribution; original not consulted",
        "T3": "DeepVol2024 identity/DOI in author CV; original pending",
        "S9": "Unpublished original covering Bitcoin and other cryptocurrencies; proposed LSTM does not count as a standalone baseline",
        "BTC_MLP_study_count": counts["MLP_baseline_studies"],
        "BTC_generic_CNN_study_count": counts["CNN_generic_baseline_studies"],
        "BTC_explicit_shallow_CNN1D_study_count": counts["CNN1D_explicit_baseline_studies"],
        "journal_index_evidence": "Independent secondary journal lists only; no individual article or Q1 verification",
        "harnet_erratum_file": summary["harnet_erratum_file"],
        "original_benchmark_changed": False, "experiment_results_modified": False,
        "new_model_fits": 0, "new_forecasts": 0, "source_sha256": stage["source_sha256"],
    }


def crypto_delivery_text(stage):
    counts = stage["counts"]
    return {
        "status": (f"{counts['bibliographic_references']} referencias; {counts['crypto_full_text_studies']} textos cripto "
                   f"({counts['BTC_full_text_studies']} con BTC) y niveles metodológicos explícitos"),
        "evidence": "references/literature_e1/" + STAGE_ID + "/summary.json",
        "limit": ("S9 es un original no publicado de varios criptoactivos; su LSTM propuesto no aumenta baselines. "
                  "HARNet conserva nivel código con errata de escalado; DeepVol conserva identidad con original pendiente. "
                  "CNN de S8 sin dimensión corroborada; CNN1D explícita, recetas exactas e indexación individual conservan sus límites."),
        "review": (f"{counts['bibliographic_references']} references; {counts['crypto_full_text_studies']} crypto full texts, "
                   f"{counts['BTC_full_text_studies']} including BTC and {counts['studies_with_non_BTC_crypto']} including other crypto. "
                   "S9 proposed LSTM is not a baseline; financial transfer sources T1/T2/T3 retain distinct evidence tiers. "
                   "No resolution of S8 CNN dimensionality, exact-recipe recurrence or individual journal indexing."),
    }
