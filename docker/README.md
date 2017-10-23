### How to setup
1. Clone repo
    ```
    git clone ...
    cd Fabrik
    ```
2. Build image
    ```
    docker build -t frontend:latest -f ./docker/Dockerfile .
    ```
3. Run
    ```
    docker-compose -f docker/docker-compose.yml up
    ```
