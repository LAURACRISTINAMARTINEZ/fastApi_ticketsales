docker run \
--rm -it \
--name api \
--memory 4g \
--cpus="1.0" \
-p 5015:5000 \
"api-343501:$1"