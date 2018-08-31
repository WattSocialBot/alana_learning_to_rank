#!/bin/sh
aws s3 cp s3://$BUCKET_LOCATION/$BUCKET_ID/ratings.csv .
