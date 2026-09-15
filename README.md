# Market Microstructure Trilogy — Paper II

**Quoting Under a Misspecified Fill Law**  
DOI: <https://doi.org/10.5281/zenodo.22759528>  
Fixed manuscript date: **15 September 2026**

The main document is `p2_quoting.tex`. Compile it with pdfLaTeX and BibTeX;
`latexmk -pdf p2_quoting.tex` performs the required passes. The bibliography,
generated `.bbl`, five publication-resolution PNG figures, and every file
needed to compile the deposited PDF are included.

The figures are deterministic and use no market data. After installing the
packages in `verification/requirements.txt`, they can be regenerated with:

    python verification/generate_manuscript_figures.py --paper 2 --output figures

`Trilogy_Citations.bib` contains the definitive BibTeX records for all three
papers. The trilogy uses a fixed star citation architecture: Paper II cites
Paper I and does not cite Paper III.

## Repository downloads

- [Final PDF](paper.pdf)
- [Complete manuscript source](Quoting_Under_a_Misspecified_Fill_Law_Source.zip)
- [Editable Overleaf project](https://www.overleaf.com/project/6aa88afd2bbfb3fcaf953981)

The repository entry point `paper.tex` is identical to `p2_quoting.tex`.
`trilogy-source-supplement.zip` is an alias of the current manuscript source archive.
`verification-suite.zip` contains the current verification scripts and figure generator.
