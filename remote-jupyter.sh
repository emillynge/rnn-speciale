#!/usr/bin/env bash
PORT=8890
LIBPATH=/home/s082768/lib:/usr/local/cuda:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/cuda-7.0/targets/x86_64-linux/lib
_PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/s082768/bin:/home/s082768/.local/bin
#chromium localhost:8887 &

CMD="echo 'logged in'; if curl localhost:$PORT; then tail -f jupyter.log; else LD_LIBRARY_PATH=$LIBPATH PATH=$_PATH /home/s082768/.local/bin/jupyter-notebook --no-browser --port=$PORT >> jupyter.log 2<&1 & tail -f jupyter.log; fi"
echo "logging in..."
ssh -p 5900  -L 8887:localhost:$PORT s082768@localhost $CMD
