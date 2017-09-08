#/bin/sh

bin_path=`dirname "${BASH_SOURCE-$0}"`
bin_path=`cd "${bin_path}"; pwd`
bin_name="http_request_demo"

curl -v "http://127.0.0.1:12345/test_http_request_demo"
