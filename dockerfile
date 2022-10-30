FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD ["python3", "./plot_digits_classification.py"]