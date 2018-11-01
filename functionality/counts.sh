#!/bin/sh

PREFIX=$1
if [ "$PREFIX" = "" ]; then
    PREFIX="."
fi
FILES=$PREFIX/*
echo "file                          \t\tnon \t\tsensitive"
for f in $FILES
do
    if [ -f $f ]; then
        CNT_0=`grep "^\ (0" $f | wc -l | cut -f1 -d' '`
        CNT_4=`grep "^\ (4" $f | wc -l | cut -f1 -d' '`
        #RATIO=`python -c "$print($CNT_4 / ($CNT_4 + $CNT_0))"`
        echo "$f\t\t$CNT_0\t\t$CNT_4"
    fi
done
