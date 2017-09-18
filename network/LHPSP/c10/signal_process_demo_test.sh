#/bin/sh

bin_name=signal_process_demo

echo "send SIGCHLD to ${bin_name}"
killall -s SIGCHLD ${bin_name}

echo "send SIGHUP to ${bin_name}"
ps aux | grep $bin_name | grep -v grep | awk '{print $2}' | xargs kill -s SIGHUP

echo "send SIGTERM to ${bin_name}"
kill -s SIGTERM `pidof ${bin_name}`


