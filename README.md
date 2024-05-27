# liveness_vit
Liveness detection using ViT model

ViT model: 
- cpu: < 10 FPS
- gpu: 60 FPS

Resnet50 model:
- cpu: < 30 fps
- gpu: < 70 FPS

## Set up NGINX

Here I have 2 servers in localhost:8091 and localhost:8092. Then I will use NGINX as an API gatway or load balancer

First, I have config to run the server using wsgi server called waitress `pip install waitress`

Then setup NGINX
    docker container run -p 8090:80 --name nginx_tutorial nginx
    docker container exec -it nginx_tutorial bash
    apt-get update
    apt-get install nano
    nano nginx.conf

The set up is as this link: https://viblo.asia/p/cau-hinh-nginx-server-nhu-the-nao-ByEZko0xZQ0

Then we modify the nginx.conf file
  ``` python
    events {
    
    }
    
    http {
    
        upstream backend_servers {
            server your_host:8091;
            server your_host:8092;
        } 
    
        server {
            listen 80;
            server_name nginx-tutorial.test;
    
            location / {
                proxy_pass http://backend_servers;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
            }       
        }
    }
```

We have to add 1 more line in the file `etc/hosts` like this

    127.0.0.1 nginx-tutorial.test
    
Reference: https://viblo.asia/p/load-balancing-voi-nginx-bWrZnVxnZxw

Then run:
      
      python src/api.py
      python src/api_2.py
      python src/test_api.py 
      
We can see that the port is changing over time -> success
