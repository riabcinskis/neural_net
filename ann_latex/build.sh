 #!/bin/bash   

rm -rf build
mkdir build

cp ./main.tex ./build/
cp ./data.txt ./build/

cd build

pdflatex main
