# IEEE Report Compilation Guide

## Prerequisites

Install LaTeX distribution:
- **macOS**: `brew install --cask mactex` or `brew install basictex`
- **Linux**: `sudo apt-get install texlive-full` (Ubuntu/Debian)
- **Windows**: Install MiKTeX or TeX Live

## Required Figures

The report references two figures that need to be created:

1. **architecture.png** - System architecture diagram
   - Location: Same directory as `IEEE_Report.tex`
   - Suggested tools: draw.io, Lucidchart, or TikZ

2. **confusion_matrix.png** - Confusion matrix
   - Location: `Ai Agent/reports/confusion_matrix.png` (already exists)
   - Copy to report directory or update path in LaTeX

## Compilation Steps

### Method 1: Using pdflatex (Recommended)

```bash
cd "/Users/meenakshsinghania04/Desktop/AI:ML Project 4"
pdflatex IEEE_Report.tex
bibtex IEEE_Report
pdflatex IEEE_Report.tex
pdflatex IEEE_Report.tex
```

### Method 2: Using latexmk (Automated)

```bash
cd "/Users/meenakshsinghania04/Desktop/AI:ML Project 4"
latexmk -pdf IEEE_Report.tex
```

### Method 3: Using Overleaf (Online)

1. Upload `IEEE_Report.tex` to Overleaf
2. Upload required figures
3. Compile using the "Recompile" button

## Figure Creation

### Architecture Diagram

Create a diagram showing:
- User Input (Text/Voice)
- Speech-to-Text Module
- Intent Classification
- Entity Extraction
- Action Execution
- Response Generation

You can use the ASCII diagram from README.md as reference.

### Confusion Matrix

The confusion matrix already exists at:
```
Ai Agent/reports/confusion_matrix.png
```

Copy it to the report directory:
```bash
cp "Ai Agent/reports/confusion_matrix.png" .
```

## Customization

### Update Author Information

Edit lines 20-30 in `IEEE_Report.tex`:
```latex
\IEEEauthorblockN{1\textsuperscript{st} Nanda P.V}
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{Your University}\\
City, Country \\
nanda@university.edu}
```

### Update Abstract

Modify the abstract section (lines 33-40) to match your specific contributions.

### Add More References

Add entries to the bibliography section (lines 200+) following IEEE format.

## Troubleshooting

### Missing Packages
If compilation fails due to missing packages:
```bash
# macOS (MacTeX)
sudo tlmgr install <package-name>

# Linux
sudo apt-get install texlive-<package-name>
```

### Figure Not Found
- Ensure figures are in the same directory as `.tex` file
- Or update paths in `\includegraphics` commands

### Bibliography Issues
- Run `bibtex` between pdflatex runs
- Ensure `.bib` file exists (or use manual bibliography as in current file)

## Output

The compiled PDF will be: `IEEE_Report.pdf`

## Tips

1. **First Compilation**: May take longer as packages are loaded
2. **Multiple Runs**: LaTeX requires multiple passes for references and citations
3. **Figure Formats**: PNG, JPG, PDF are all supported
4. **Page Limit**: IEEE conference papers typically have 6-8 page limit

