#!/bin/bash

# Argument 1 is database name, argument 2 is path to dump file to be restored
pg_restore -U $USER -d $1 $2