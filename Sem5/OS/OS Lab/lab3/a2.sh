echo "Enter the Numbers (space-separated):"
read -a arr

sum=0
for num in "${arr[@]}"
do
    sum=$((sum + num))
done

echo "Sum: $sum"
