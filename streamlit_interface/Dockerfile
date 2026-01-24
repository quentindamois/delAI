FROM python:3.13

WORKDIR ./streamlit_interface


COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


EXPOSE 8051

CMD [ "streamlit", "run", "main.py"]