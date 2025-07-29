echo "Enter N:"
read n
num=1
while [ $n -ne 0 ]
do
	echo "$num"
	num=$(( num + 2 ))
	n=$(( n - 1 ))
done