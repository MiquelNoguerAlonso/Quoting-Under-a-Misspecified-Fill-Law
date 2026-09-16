# Market Microstructure Trilogy — Paper II

**Quoting Under a Misspecified Fill Law**  
DOI: <https://doi.org/10.5281/zenodo.22759528>  
Fixed manuscript date: **15 September 2026**

The main document is `p2_quoting.tex`. Compile it with pdfLaTeX and BibTeX;
`latexmk -pdf p2_quoting.tex` performs the required passes. The bibliography,
generated `.bbl`, five publication-resolution PNG figures, and every file
needed to compile the deposited PDF are included.

The included verification suite is deterministic and uses no market data.
Install its pinned Python dependencies and run all 41 check groups with:

    python -m pip install -r verification/requirements.txt
    python verification/verify_theory.py

The command stops on a failed assertion and rewrites
`verification/verification_results.json`. The five figures can then be
regenerated with:

    python verification/generate_manuscript_figures.py --paper 2 --output figures

Recompile after regeneration to reproduce the deposited PDF.

`Trilogy_Citations.bib` contains the definitive BibTeX records for all three
papers. The trilogy uses a fixed star citation architecture: Paper II cites
Paper I and does not cite Paper III.

`MANIFEST_SHA256.txt` inventories every other file in the source archive. On
systems with GNU Coreutils, verify it before regeneration with
`sha256sum -c MANIFEST_SHA256.txt`.

The final release audit is recorded in `SUBMISSION_REVIEW_2026-09-15.md`.
