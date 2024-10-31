for file in logs/gs*
do
    echo ==================================================
    echo $file
    echo
    cat $file | python utils/comp.py
    echo
done