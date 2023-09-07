python evaluate.py bs
for N in 500 750 1000 1250 1500 1750 2000
do
    python evaluate.py ef $1 --n $N
    python evaluate.py ws $1 --n $N
    python evaluate.py tr $1 --n $N --delta 0.001
    python evaluate.py tr $1 --n $N --delta 0.01
done