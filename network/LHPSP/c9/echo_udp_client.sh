#/bin/sh

if [ "$#" -lt "2" ]; then
	echo "usage: $0 ip port"
	exit 1
fi

ip="$1"
port="$2"

nc -u "$ip" "$port"
