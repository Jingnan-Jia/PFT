#!/bin/bash

# Use a fifo as an ad-hoc semaphore.
# Only one script can create it at a time, the rest
# will wait.
while ! mkfifo /tmp/script-fifo 2> /dev/null
do
        sleep 1
done

COUNT=0
[ -f /home/jjia/data/lung_function/lung_function/scripts/script2.sh ] && read COUNT < /home/jjia/data/lung_function/lung_function/scripts/script2.sh
((COUNT++))
echo $COUNT > /home/jjia/data/lung_function/lung_function/scripts/script2.sh
rm -f /tmp/script-fifo

echo "script has been called $COUNT times"

echo 'finish'
