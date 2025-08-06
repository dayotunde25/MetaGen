#!/bin/bash
echo "Starting Celery worker for AIMetaHarvest..."
echo ""
echo "Make sure Redis is running before starting the worker!"
echo ""
echo "Using solo pool for better compatibility..."
echo ""
celery -A celery_app.celery_app worker --loglevel=info --pool=solo
