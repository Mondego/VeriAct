.PHONY: all clean intro warnings more tool tips

all: intro warnings more tool tips

clean: 	
	rm -f *.aux
	rm -f *.log
	rm -f *.ps
	rm -f *.pdf
	rm -f *.dvi
	rm -f *.out

intro: 1_intro_jml.pdf

1_intro_jml.pdf: 1_intro_jml.tex
	latex 1_intro_jml.tex 
	dvips -Ppdf -G0 1_intro_jml.dvi
	ps2pdf 1_intro_jml.ps
	psnup -4 -l 1_intro_jml.ps > 1_intro_jml4up.ps
	ps2pdf 1_intro_jml4up.ps
	rm 1_intro_jml.dvi

tool: 2_tool.pdf

2_tool.pdf: 2_tool.tex
	latex 2_tool.tex 
	dvips -Ppdf -G0 2_tool.dvi
	ps2pdf 2_tool.ps
	psnup -4 -l 2_tool.ps > 2_tool4up.ps
	ps2pdf 2_tool4up.ps
	rm 2_tool.dvi

warnings: 3_warnings.pdf

3_warnings.pdf: 3_warnings.tex
	latex 3_warnings.tex 
	dvips -Ppdf -G0 3_warnings.dvi
	ps2pdf 3_warnings.ps
	psnup -4 -l 3_warnings.ps > 3_warnings4up.ps
	ps2pdf 3_warnings4up.ps
	rm 3_warnings.dvi

tips: 4_tips.pdf

4_tips.pdf: 4_tips.tex
	latex 4_tips.tex 
	dvips -Ppdf -G0 4_tips.dvi
	ps2pdf 4_tips.ps
	psnup -4 -l 4_tips.ps > 4_tips4up.ps
	ps2pdf 4_tips4up.ps
	rm 4_tips.dvi

more: 5_more_jml.pdf

5_more_jml.pdf: 5_more_jml.tex
	latex 5_more_jml.tex 
	dvips -Ppdf -G0 5_more_jml.dvi
	ps2pdf 5_more_jml.ps
	psnup -4 -l 5_more_jml.ps > 5_more_jml4up.ps
	ps2pdf 5_more_jml4up.ps
	rm 5_more_jml.dvi
