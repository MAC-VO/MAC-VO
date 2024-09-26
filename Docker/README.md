# Docker Setup for Conda Environment

This Docker setup provides a template for creating a Docker image with a Miniconda environment installed under a user-specified username.

## Prerequisites

- Docker must be installed on your machine. You can download and install Docker from [Docker's official website](https://www.docker.com/get-started).

## Dockerfile with your own user name (Optional)

Before building the image, you may want to customize the username under which Miniconda will be installed. To do this, open the `Dockerfile` in your favorite text editor and look for the line:

```dockerfile
ARG USERNAME=user
```

Replace `user` with your desired username. Save the changes.

### Build

To build the Docker image, run the following command from the directory containing the Dockerfile:

```bash
docker build --no-cache -t your-image-name:latest .
```

Replace `your-image-name` with a name of your choice for the Docker image.

## Docker file with root (Optional)

To build the Docker image, run the following command from the directory containing the Dockerfile:

```bash
docker build --no-cache -t -f DockerfileRoot your-image-name:latest .
```

Replace `your-image-name` with a name of your choice for the Docker image.


## Run container

* Replace `your-image-name` with a name of your choice for the Docker image.

```bash
docker run -td --net=host --ipc=host \
    --name="AirVIO_Dev" \
    --gpus=all \
    -e "DISPLAY=\$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix\$DISPLAY \
    -v "/mnt/c/DevHome:/code"\
    -v "/mnt/d/Data:/data"\
    -e "XAUTHORITY=\$XAUTH" \
    -e ROS_IP=127.0.0.1 \
    --cap-add=SYS_PTRACE \
    -v /etc/group:/etc/group:ro \
    yutianchen/your-image-name:latest bash
```
