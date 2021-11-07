wget -nc http://vis-www.cs.umass.edu/lfw/lfw.tgz

echo 'creating ../generated if it does not exist already'
mkdir -p ../generated

printf '\nuntaring dataset to ../generated'
tar -xf lfw.tgz -C ../generated
printf '\n'

printf 'deleting lfw.tgz'
rm -rf lfw.tgz
printf '\n'
