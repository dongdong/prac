#!/bin/sh

action="$1"
uname=`uname`
jar=out/artifacts/WeatherData_jar/WeatherData.jar

MaxTemperature() {
	mainclass=com.dd.demo.mapreduce.MaxTemperature
	input=~/data/hadoopbook/input/ncdc/sample.txt
	output=~/output/MaxTemperature

	hadoop fs -rm -r -f ${output}
	hadoop jar ${jar} ${mainclass} ${input} ${output}

	if [ "$?" = 0 ]; then
		echo "run SUCCESS!"
		hadoop fs -ls ${output}
		hadoop fs -cat ${output}/part-r-00000
	fi
}

FileSystemCat() {
	mainclass=com.dd.demo.hdfs.FileSystemCat
	input=~/data/hadoopbook/input/ncdc/sample.txt
	
	echo "FileCat" "$input"
	hadoop jar ${jar} ${mainclass} ${input}
}

FileCopyWithProgress() {
	mainclass=com.dd.demo.hdfs.FileCopyWithProgress
	input=./pom.xml
	outputdir=~/data/output/FileCopy
	output=${outputdir}/copiedFile

	hadoop fs -mkdir -p ${outputdir}
	hadoop fs -rm -f ${output}
	
	echo "FileCopy" "$input" "$output"
	hadoop jar ${jar} ${mainclass} ${input} ${output}
	
	if [ "$?" = 0 ]; then
		echo "copy SUCCESS!"
		hadoop fs -ls ${outputdir}
		hadoop fs -cat ${output}
	fi
}

Main() {
	if [ "${uname}" = "Darwin" ]; then
		echo "MacOS"
		zip -d ${jar} META-INF/LICENSE
	else
		echo "Linux"
	fi

	if [ "$action" = "MaxTemperature" ]; then
		MaxTemperature
	elif [ "$action" = "FileCat" ]; then
		FileSystemCat
	elif [ "$action" = "FileCopy" ]; then
		FileCopyWithProgress
	else
		echo "Unknown action!"
	fi
	
	echo ""
}

Main "$@"
