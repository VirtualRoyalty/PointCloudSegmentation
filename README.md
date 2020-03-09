# PointCloudSegmentation

---
Project description

<br>

## How to dockerize this:
---
- In *base-notebook/* folder start Docker and build an image:
  `$ docker build -t jupyter .`
- After that you can verify a successful build by running: `$ docker images`
- Then start container by running:<br><br>
  `$  docker run -it --rm -p 8888:8888  -v /path/to/obstacle-detection:/home/jovyan/work jupyter` <br><br>
  **NOTE:**  on Windows  you need to convert your path into a quasi-Linux format (*e.g. //c/path/to/obstacle-detection*). More details [here](https://medium.com/@kale.miller96/how-to-mount-your-current-working-directory-to-your-docker-container-in-windows-74e47fa104d7) <br>
  Also, if you want to use drive *D:/* you need to check whether it is mounted or not and if not mount it manually. More details [here](http://support.divio.com/en/articles/646695-how-to-use-a-directory-outside-c-users-with-docker-toolbox-docker-for-windows) if you use Docker toolbox <br><br>
- After correct running you will see URL to access jupyter, e.g.: <br><br>
             *httр://127.0.0.1:8888?token=0cccd15e74216ed2dbe681738ed0f9c78bf65515e94f27a8*<br><br>
- To access jupyter you need to go for **Docker IP**:8888?token=xxxx... <br>( e.g.  httр://192.168.99.100:8888/?token=0cccd15e74216ed2dbe681738ed0f9c78bf65515e94f27a8)<br><br>
- To enter a docker container run `$ docker exec -it *CONTAINER ID* bash` (find out ID by running `$ docker ps`)
=======
