Lightweight LaTeX compile helper
================================

Purpose
-------
This repository includes several LaTeX samples and the full ACM template. Running the top-level `Makefile` in `report/` compiles additional template files and may require many TeX packages.

If you only want to compile the project report `samples/sample-acmcp.tex` to PDF quickly, use the small helper script `build_sample.sh` which invokes `pdflatex` directly on the sample file and places the output in `report/output/`.

Usage
-----
Ensure `pdflatex` is installed (TeX Live or similar). Then run from the project root or the `report/` folder:

```bash
cd report
./build_sample.sh
```

Output
------
The generated PDF will be written to `report/output/sample-acmcp.pdf`.

Notes
-----
- The script deliberately avoids running the full `Makefile` (which builds `acmart.cls` and other artifacts).
- If the PDF compilation fails due to missing LaTeX packages, install a fuller TeX Live distribution (e.g., `texlive-latex-recommended` or `texlive-full` on Debian/Ubuntu).
- If you prefer an HTML version, consider installing `pandoc` and converting the `.tex` file to HTML (not all LaTeX constructs convert cleanly).
