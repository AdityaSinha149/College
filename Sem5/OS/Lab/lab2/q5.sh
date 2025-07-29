read f1
read f2
f3="f3.txt"
cat $f1 > f3
cat $f2 >> f3
sort f3 -u -o f3