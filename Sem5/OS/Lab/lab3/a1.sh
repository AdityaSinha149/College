echo "Enter String"
read str
len=${#str}
p=1
for(( i=0;i<len;i++ )) do
    if [ ${str:i:1} != ${str:len-i-1:1} ]; then
        p=0
        break
    fi
done
if [ $p -eq 1 ]; then
    echo "$str is a Palindrome"
else 
    echo "$str is not a Palindrome"
fi