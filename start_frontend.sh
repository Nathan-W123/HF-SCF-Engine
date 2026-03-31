#!/bin/bash
# Start the React frontend
cd "$(dirname "$0")/frontend"

echo "Installing frontend dependencies..."
npm install

echo ""
echo "Starting HF-SCF Calculator frontend on http://localhost:3000"
echo ""
npm start
