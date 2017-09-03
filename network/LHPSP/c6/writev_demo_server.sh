#/bin/sh

bin_path=`dirname "${BASH_SOURCE-$0}"`
bin_path=`cd "${bin_path}"; pwd`
bin_name="writev_demo"

${bin_path}/${bin_name} "127.0.0.1" 12346 writev_demo.c
