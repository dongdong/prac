#/bin/sh

bin_path=`dirname "${BASH_SOURCE-$0}"`
bin_path=`cd "${bin_path}"; pwd`
bin_name="euid_demo"

test_file="ROOTFILE"

cd ${bin_path}

rm -rf ${bin_name}
make ${bin_name}
./${bin_name}

sudo chown root:root ${bin_name}
sudo chmod +s ${bin_name}
./${bin_name}

#rm -rf ${bin_name} ${test_file} 

cd -
