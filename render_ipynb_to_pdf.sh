jupyter nbconvert --to latex $1.ipynb
sed -r -i 's/documentclass\[9pt\]\{article\}/documentclass[9pt]{extarticle}/' $1.tex
sed -r -i 's/geometry\{verbose,tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in}/geometry{verbose,tmargin=0.35in,bmargin=0.35in,lmargin=0.15in,rmargin=0.15in}/' $1.tex
pdflatex $1
rm $1.tex; rm $1.out; rm $1.aux; rm $1.log
