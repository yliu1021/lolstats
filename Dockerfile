FROM python:3.10.5-bullseye
RUN useradd -ms /bin/bash lolstats
USER lolstats
WORKDIR /home/lolstats
ENV PATH "$PATH:/home/lolstats/.local/bin"
RUN pip install "flask[async]" motor python-dotenv torch aiohttp numpy
COPY . .
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
