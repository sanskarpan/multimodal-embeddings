#!/bin/bash

# Weaviate Setup Script
# This script helps manage the Weaviate Docker container

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[+]${NC} $1"
}

print_error() {
    echo -e "${RED}[!]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[*]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_status "Docker is installed and running"
}

# Check if .env file exists
check_env() {
    if [ ! -f .env ]; then
        print_error ".env file not found. Please create one with your OPENAI_API_KEY"
        exit 1
    fi
    
    if ! grep -q "OPENAI_API_KEY=" .env; then
        print_error "OPENAI_API_KEY not found in .env file"
        exit 1
    fi
    
    print_status ".env file configured"
}

# Start Weaviate
start_weaviate() {
    print_status "Starting Weaviate..."
    docker-compose up -d
    
    print_status "Waiting for Weaviate to be ready..."
    sleep 5
    
    # Wait for Weaviate to be healthy
    max_attempts=30
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8080/v1/.well-known/ready &> /dev/null; then
            print_status "Weaviate is ready!"
            docker-compose ps
            echo ""
            echo -e "${GREEN}✓ Weaviate is running on http://localhost:8080${NC}"
            echo -e "${GREEN}✓ gRPC endpoint is available on localhost:50051${NC}"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    print_error "Weaviate failed to start properly"
    docker-compose logs
    exit 1
}

# Stop Weaviate
stop_weaviate() {
    print_status "Stopping Weaviate..."
    docker-compose down
    print_status "Weaviate stopped"
}

# Restart Weaviate
restart_weaviate() {
    stop_weaviate
    start_weaviate
}

# Show Weaviate status
status_weaviate() {
    docker-compose ps
    echo ""
    
    if curl -s http://localhost:8080/v1/.well-known/ready &> /dev/null; then
        echo -e "${GREEN}✓ Weaviate is running on http://localhost:8080${NC}"
        
        # Show meta information
        echo ""
        print_status "Weaviate meta information:"
        curl -s http://localhost:8080/v1/meta | python3 -m json.tool
    else
        echo -e "${RED}✗ Weaviate is not responding${NC}"
    fi
}

# Show logs
logs_weaviate() {
    docker-compose logs -f
}

# Clean up everything (including volumes)
clean_weaviate() {
    print_warning "This will remove all Weaviate data. Are you sure? (yes/no)"
    read -r response
    if [ "$response" = "yes" ]; then
        print_status "Cleaning up Weaviate..."
        docker-compose down -v
        print_status "All Weaviate data has been removed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Main script
case "$1" in
    start)
        check_docker
        check_env
        start_weaviate
        ;;
    stop)
        stop_weaviate
        ;;
    restart)
        check_docker
        check_env
        restart_weaviate
        ;;
    status)
        status_weaviate
        ;;
    logs)
        logs_weaviate
        ;;
    clean)
        clean_weaviate
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|clean}"
        echo ""
        echo "Commands:"
        echo "  start   - Start Weaviate containers"
        echo "  stop    - Stop Weaviate containers"
        echo "  restart - Restart Weaviate containers"
        echo "  status  - Show Weaviate status and meta information"
        echo "  logs    - Show and follow Weaviate logs"
        echo "  clean   - Stop and remove all Weaviate data (dangerous!)"
        exit 1
        ;;
esac


