# trip-advice-agent-backend

## How to build and run

1. Build image

```bash
docker build -t trip-advice-agent-backend .
```

2. Run container

```bash
docker run -it --env-file=.env --rm --name trip-advice-agent-backend -p 8081:80 trip-advice-agent-backend
```

## push to docker hub

```bash
docker tag trip-advice-agent-backend $USER/trip-advice-agent-backend:latest

docker push $USER/trip-advice-agent-backend:latest
```
