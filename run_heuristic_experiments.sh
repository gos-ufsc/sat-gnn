python evaluate.py bs $2
for N in 500 750 1000 1250 1500 1750 2000
do
    python evaluate.py ef $1 --n $N $2
    python evaluate.py ws $1 --n $N $2
    python evaluate.py tr $1 --n $N --delta 1 $2
    python evaluate.py tr $1 --n $N --delta 2 $2
    python evaluate.py tr $1 --n $N --delta 5 $2
done