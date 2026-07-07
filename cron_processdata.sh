#!/bin/bash
#Script for running processdata-lxdlwagpu09.sh as a cronjob
echo "===== Processing started at $(date) =====" >> /home/ubuntu/kp/logs/processdata-$(date +%Y%m%d).log
/home/ubuntu/kp/lwa-cosmic-rays/processdata-lxdlwagpu09.sh "$(date +%Y%b%-d)"  2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0 }'  >> /home/ubuntu/kp/logs/processdata-$(date +%Y%m%d).log
