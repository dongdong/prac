#/bin/sh

bin_path=`dirname "${BASH_SOURCE-$0}"`
bin_path=`cd "${bin_path}"; pwd`
bin_name="nonblock_connect_demo"

echo "start server in background"
nc -l 127.0.0.1 12345 &

sleep 1

echo "start client"
${bin_path}/${bin_name} "127.0.0.1" 12345
