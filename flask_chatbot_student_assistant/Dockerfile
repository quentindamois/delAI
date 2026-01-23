FROM python:3.13

WORKDIR ./flask_app_llm


COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


EXPOSE 5000

CMD [ "python", "-m", "flask", "run", "--host=0.0.0.0"]