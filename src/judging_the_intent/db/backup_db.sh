#!/bin/bash

# Argument is name of the database to backup
timestamp=$(date +"%Y%m%d_%H%M%S")
pg_dump -U "$USER" -F c $1 -f "$1_backup_$timestamp.dump"