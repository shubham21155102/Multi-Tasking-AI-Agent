#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Multilingual Note-Taking Agent Docker Setup ===${NC}"
echo -e "${BLUE}This script will help you run the application using Docker${NC}"
echo ""

# Check if Docker is installed
if ! [ -x "$(command -v docker)" ]; then
  echo -e "${RED}Error: Docker is not installed.${NC}" >&2
  echo "Please install Docker from https://docs.docker.com/get-docker/"
  exit 1
fi

# Check if Docker Compose is installed
if ! [ -x "$(command -v docker-compose)" ]; then
  echo -e "${RED}Error: Docker Compose is not installed.${NC}" >&2
  echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
  exit 1
fi

echo -e "${GREEN}Step 1: Building Docker images...${NC}"
docker-compose build

echo -e "${GREEN}Step 2: Starting the containers...${NC}"
docker-compose up -d

echo -e "${GREEN}Success! Your application is now running.${NC}"
echo -e "Frontend: ${BLUE}http://localhost${NC}"
echo -e "Backend: ${BLUE}http://localhost:8000${NC}"
echo ""
echo -e "To stop the application, run: ${BLUE}docker-compose down${NC}"
echo -e "To view logs, run: ${BLUE}docker-compose logs -f${NC}"
echo ""
echo -e "${GREEN}Thank you for using Multilingual Note-Taking Agent!${NC}"