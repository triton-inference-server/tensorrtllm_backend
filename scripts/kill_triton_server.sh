ps -ef | grep 'tritonserver' | grep -v grep | awk '{print $2}' | xargs -r kill -9
