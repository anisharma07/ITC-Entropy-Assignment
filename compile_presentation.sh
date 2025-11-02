#!/bin/bash
# Script to properly compile the presentation with TOC

echo "Compiling presentation.tex..."
echo "=============================="

# Clean old auxiliary files
echo "Step 1: Cleaning old auxiliary files..."
rm -f presentation.aux presentation.toc presentation.nav presentation.snm presentation.out presentation.log presentation.fdb_latexmk presentation.fls presentation.vrb

# First compilation
echo "Step 2: First compilation (generating auxiliary files)..."
pdflatex -interaction=nonstopmode presentation.tex > /dev/null 2>&1

# Second compilation
echo "Step 3: Second compilation (incorporating TOC)..."
pdflatex -interaction=nonstopmode presentation.tex > /dev/null 2>&1

# Third compilation (optional, for stability)
echo "Step 4: Third compilation (final polish)..."
pdflatex -interaction=nonstopmode presentation.tex

echo ""
echo "=============================="
echo "Compilation complete!"
echo "Output: presentation.pdf"
echo ""
echo "TOC file size: $(ls -lh presentation.toc | awk '{print $5}')"
echo "Total pages: $(pdfinfo presentation.pdf 2>/dev/null | grep Pages | awk '{print $2}')"
