#!/bin/sh

uname=`uname`
jar=out/artifacts/WeatherData_jar/WeatherData.jar

if [ "${uname}" = "Darwin" ]; then
	echo "MacOS"
	zip -d ${jar} META-INF/LICENSE
else
	echo "Linux"
fi

output=~/output/MaxTemperature
hadoop fs -rm -r -f ${output}

hadoop jar ${jar} com.dd.demo.mapreduce.MaxTemperature ~/data/hadoopbook/input/ncdc/sample.txt ${output}

if [ "$?" = 0 ]; then
	echo "run SUCCESS!"
	hadoop fs -ls ${output}
fi
