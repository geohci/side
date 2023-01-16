#!/usr/bin/env bash
# restart API endpoint with new code

# these can be changed but most other variables should be left alone
APP_LBL='api-endpoint'  # descriptive label for endpoint-related directories
REPO_LBL='side'  # directory where repo code will go
GIT_CLONE_HTTPS='https://github.com/geohci/side'  # for `git clone`
GIT_BRANCH="main"

# derived paths
ETC_PATH="/etc/${APP_LBL}"  # app config info, scripts, ML models, etc.
TMP_PATH="/tmp/${APP_LBL}"  # store temporary files created as part of setting up app (cleared with every update)
LIB_PATH="/var/lib/${APP_LBL}"  # where virtualenv will sit

# clean up old versions
rm -rf ${TMP_PATH}
mkdir -p ${TMP_PATH}

git clone --branch ${GIT_BRANCH} ${GIT_CLONE_HTTPS} ${TMP_PATH}/${REPO_LBL}

# reinstall virtualenv
rm -rf ${LIB_PATH}/p3env
echo "Setting up virtualenv..."
python3 -m venv ${LIB_PATH}/p3env
source ${LIB_PATH}/p3env/bin/activate

echo "Installing repositories..."
pip install wheel
pip install gunicorn[gevent]
pip install -r ${TMP_PATH}/${REPO_LBL}/requirements.txt

echo "Setting up ownership..."  # makes www-data (how nginx is run) owner + group for all data etc.
chown -R www-data:www-data ${ETC_PATH}
chown -R www-data:www-data ${LIB_PATH}
chown -R www-data:www-data ${TMP_PATH}

echo "Copying configuration files..."
cp -r ${TMP_PATH}/${REPO_LBL}/verify_wikipedia ${ETC_PATH}
cp ${TMP_PATH}/${REPO_LBL}/api_config/gunicorn.conf.py ${ETC_PATH}
cp ${TMP_PATH}/${REPO_LBL}/api_config/flask_config.yaml ${ETC_PATH}
cp ${TMP_PATH}/${REPO_LBL}/api_config/model.service /etc/systemd/system/
cp ${TMP_PATH}/${REPO_LBL}/api_config/model.nginx /etc/nginx/sites-available/model
if [[ -f "/etc/nginx/sites-enabled/model" ]]; then
    unlink /etc/nginx/sites-enabled/model
fi
ln -s /etc/nginx/sites-available/model /etc/nginx/sites-enabled/

echo "Enabling and starting services..."
systemctl enable model.service  # uwsgi starts when server starts up
systemctl daemon-reload  # refresh state

systemctl restart model.service  # start up uwsgi
systemctl restart nginx  # start up nginx