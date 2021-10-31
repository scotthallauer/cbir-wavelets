run:
	python3 src/app.py
setup: requirements.txt
	pip3 install -r requirements.txt
clean:
	rm -rf src/__pycache__