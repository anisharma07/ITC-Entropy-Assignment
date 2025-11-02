# LaTeX Presentation Compilation Guide

## Problem Fixed
The table of contents (TOC) was appearing on the first compile but disappearing on subsequent compilations.

## Root Cause
The issue was caused by:
1. Corrupted or empty auxiliary files (`.toc`, `.nav`, `.snm`)
2. LaTeX/Beamer requires multiple compilation passes to properly generate TOC

## Solution Applied
1. Added `[allowframebreaks]` option to the TOC frame in `presentation.tex`
2. Created a compilation script that properly handles multiple passes

## How to Compile

### Option 1: Use the Compilation Script (Recommended)
```bash
./compile_presentation.sh
```

This script will:
- Clean old auxiliary files
- Run pdflatex three times (ensuring TOC is properly generated)
- Display compilation status

### Option 2: Manual Compilation
If you prefer to compile manually, always run pdflatex **at least twice**:

```bash
pdflatex presentation.tex
pdflatex presentation.tex
```

### Option 3: Using latexmk (if available)
```bash
latexmk -pdf presentation.tex
```

## Important Notes

1. **Always compile twice**: LaTeX needs two passes to properly generate the TOC:
   - First pass: Generates auxiliary files with section information
   - Second pass: Uses those files to create the actual TOC

2. **If TOC disappears again**: Delete all auxiliary files and recompile:
   ```bash
   rm -f presentation.aux presentation.toc presentation.nav presentation.snm presentation.out
   pdflatex presentation.tex
   pdflatex presentation.tex
   ```

3. **Auxiliary files to watch**:
   - `presentation.toc` - Should be ~298 bytes (contains TOC data)
   - `presentation.nav` - Should be ~3.2KB (contains navigation data)
   - `presentation.snm` - Can be empty (snippet markers)

## Verification

After compilation, check that the TOC file has content:
```bash
ls -lh presentation.toc
cat presentation.toc
```

You should see section entries like:
```
\beamer@sectionintoc {1}{Introduction}{3}{0}{1}
\beamer@sectionintoc {2}{Methodology}{5}{0}{2}
...
```

## Current Status
✅ TOC is now working correctly
✅ Verified across multiple compilations
✅ Compilation script provided for convenience
