# Quoting Under a Misspecified Fill Law

**Miquel Noguer Alonso**  
Artificial Intelligence Finance Institute (AIFI)

A theory of order-book queues, adverse selection, partial fills, cancellation latency, inventory control, and performance certificates under fill-law misspecification.

- DOI: [10.5281/zenodo.22759528](https://doi.org/10.5281/zenodo.22759528)
- Overleaf: [editable project](https://www.overleaf.com/project/6aa88afd2bbfb3fcaf953981)
- Manuscript: [`paper.pdf`](paper.pdf)
- LaTeX: [`paper.tex`](paper.tex)

## Repository contents

This private repository is part 2 of the *Market Microstructure Trilogy*. It contains the reviewed manuscript, its LaTeX source, and the corresponding source and verification archive. The manuscript uses author-year citations and includes a table of contents.

## Build

The manuscript was built with pdfLaTeX. For Papers II and III, run BibTeX between LaTeX passes.

```bash
latexmk -pdf paper.tex
```

## Verification status

The released PDF was reproduced from the included source on 15 September 2026. The Overleaf build completed with zero errors and zero warnings. Numerical and symbolic checks are contained in the accompanying verification archive.

## Scope

The guarantees in the paper are conditional on the declared models, information sets, and uncertainty bounds. Synthetic calculations verify the stated identities and certificates; they do not claim live-market profitability.
