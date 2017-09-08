#/bin/sh

bin_path=`dirname "${BASH_SOURCE-$0}"`
bin_path=`cd "${bin_path}"; pwd`
bin_name="tee_demo"
file_name="test_tee.txt"

${bin_path}/${bin_name} ${file_name}

echo "contents in file:"

cat ${file_name}
