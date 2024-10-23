# This script is to divide the dataset into 80% 20%
mkdir ../data
mv /root/.cache/kagglehub/datasets/jessicali9530/caltech256/versions/2/256_ObjectCategories ../data

base_folder="../data/divided/"

mkdir -p "$base_folder"

for dir in "../data/256_ObjectCategories/"*
do

  subdir=$(basename "$dir")

  mkdir -p "$base_folder$subdir"

  count=$(ls "$dir" | wc -l)

  tenpercent=$(expr $count '*' 20 '/' 100)

  ls "$dir" | shuf -n "$tenpercent" | xargs -I {} mv "$dir"/{} "$base_folder$subdir"

done
