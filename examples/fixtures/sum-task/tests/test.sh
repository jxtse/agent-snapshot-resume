#!/bin/bash
mkdir -p /logs/verifier
if [ "$(cat /app/out.txt)" = "5" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
