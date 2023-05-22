FROM python:3.8
EXPOSE 8080
WORKDIR /app
COPY . ./
RUN pip install -r requirements.txt
CMD streamlit run --server.port 8080 --server.enableCORS false app.py
