FROM python:3.10

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["voila", "--no-browser", "--Voila.ip=0.0.0.0", "view.ipynb"]