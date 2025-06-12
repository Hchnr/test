# set -ex


ls -l /share/project/hcr/models/wenxinyiyan/paddle_internal/ERNIE-45-Turbo | awk -F ' ' '{print $9}' | sort > done.txt
bcecmd bos ls  bos:/ernie45t/paddle_internal/ERNIE-45-Turbo/ | grep -v git | awk -F ' ' '{print $5}' | sort > all.txt
comm -23 all.txt done.txt  > todo.txt

n_all=`cat all.txt| wc -l`
n_todo=`cat todo.txt | wc -l`
n_done=$(( n_all - n_todo ))
echo "DONE: $n_done/$n_all, TODO: $n_todo/$n_all"

while read -r line; do
    echo "Downloading $line"
    if ps -ef | grep "$line" | grep -v grep > /dev/null 2>&1; then
        echo "$line running"
    else
        echo "$line starting"
        rm "log/$line.log"
        nohup bcecmd bos cp bos:/ernie45t/paddle_internal/ERNIE-45-Turbo/$line "/share/project/hcr/models/wenxinyiyan/paddle_internal/ERNIE-45-Turbo/" > "log/$line.log" &
    fi
    
done < todo.txt
