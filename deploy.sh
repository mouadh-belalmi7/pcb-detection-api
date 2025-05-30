#!/bin/bash
echo "🚀 Starting PCB Detection API..."
docker-compose down
docker-compose build
docker-compose up -d
echo "✅ Deployment complete! API is running at http://localhost:5000"