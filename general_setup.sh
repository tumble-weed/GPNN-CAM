apt-get install gcc
apt-get install unzip -y
virtualenv .
source bin/activate
pip install -r requirements.txt
bash setup_rclone.sh
