all:
	latexmk --shell-escape -pdf classifier_tests.tex

clean:
	rm -f *.log *.aux *.dvi *.bbl *.blg *.toc *.out *latexmk *fls *pdf
