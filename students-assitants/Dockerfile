
# Solution to install python from Daniel Campos Olivares
# from "Use Python and Node.js in the same Dockerfile and create one image that I cloud use both"
# link : https://stackoverflow.com/questions/71197350/use-python-and-node-js-in-the-same-dockerfile-and-create-one-image-that-i-cloud



FROM python:3.13-alpine

WORKDIR ./team_chatbot


COPY . .

RUN apk update && apk add nodejs npm


RUN npx @microsoft/teams.cli --version


RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install uv


EXPOSE 3978
EXPOSE 3979

CMD ["uv", "run", "start"]
