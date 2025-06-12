set -ex

rm -rf ./log/download* 
while true
do
    echo "Downloading bcecmd models"
    stamp=`date`
    ./download.sh >  "log/download-$stamp.log"
    sleep 30
done
