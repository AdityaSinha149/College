echo "Enter the number:"
read num
num=$(( num & 1 ))
if [ $num -eq 0 ]; then 
    echo "Even"
else
    echo "Odd"
fi