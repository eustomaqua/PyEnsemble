# Set up the base image
FROM --platform=linux/amd64 ubuntu:latest

# To install Miniconda
RUN apt-get update && apt-get install -y wget
RUN apt-get update && apt-get install -y vim
RUN apt-get update && apt-get install -y git
