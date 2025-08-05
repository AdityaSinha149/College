echo "For Ax^2 + Bx + C = 0, enter A, B, C:"
read a b c
d=$(( b*b - 4*a*c ))

if [ $d -gt 0 ]; then
    condition=1
elif [ $d -eq 0 ]; then
    condition=0
else
    condition=-1
fi

case $condition in
  1)  # d > 0
      r1=$(echo "scale=2; (-$b + sqrt($d)) / (2 * $a)" | bc -l)
      r2=$(echo "scale=2; (-$b - sqrt($d)) / (2 * $a)" | bc -l)
      echo "Real & distinct roots: $r1 and $r2"
      ;;
  0)  # d = 0
      r=$(echo "scale=2; (-1)*$b / (2 * $a)" | bc -l)
      echo "Real & equal root: $r"
      ;;
 -1)  # d < 0
      d_pos=$(( -d ))
      real=$(echo "scale=2; (-1)*$b / (2 * $a)" | bc -l)
      imag=$(echo "scale=2; sqrt($d_pos) / (2 * $a)" | bc -l)
      echo "Complex roots: $real+${imag}i and $real-${imag}i"
      ;;
esac
