#/bin/sh

bin_path=`dirname "${BASH_SOURCE-$0}"`
bin_path=`cd "${bin_path}"; pwd`
bin_name="sendfile_demo"

${bin_path}/${bin_name} "127.0.0.1" 12347 sendfile_demo.c
