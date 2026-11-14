# Citation coverage and primary-source roles

Observed on 2026-11-14 UTC (2026-11-13 Bogotá). This is a retrospective audit of the current catalogue, not a replacement for its original native searches.

`citation_coverage.csv` and `openalex_access_receipt.json` contain ten systems linked to nine distinct studies. All nine official OpenAlex API requests failed with a proxy403 response. Citation counts remain missing; missing is never converted to zero. Web access to the official OpenAlex API/UI and Semantic Scholar API also failed, as recorded in `web_access_receipt.json`.

One publisher search result exposed a Nature citation count for the autoencoder paper. It is retained only inside the web receipt, not mixed with Scopus or OpenAlex, not used for a score, and not represented as a newly retrieved uniform bibliometric snapshot. The old Scopus counts are not modified.

`fetch_openalex.py` reproduces the attempted coherent-provider retrieval. If a later run succeeds, each DOI/title identity must be checked before any score can use the count; arXiv and journal versions must not silently be merged or double-counted.

`current_primary_adjudication.json` derives source roles from the existing verified extraction. It preserves existing read levels and locators without claiming a new full-text reading or human review. All nine studies support architecture transfer and context; none is labelled direct comparable evidence for BTC daily sigma7. Wang2016 is an explicit foundational exception outside2017–2026; FCN and ResNet share that one study. Final E2 test scores were not read or used to choose these sources.
