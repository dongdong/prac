#/bin/sh

bin_path=`dirname "${BASH_SOURCE-$0}"`
bin_path=`cd "${bin_path}"; pwd`

nc 127.0.0.1 12347
