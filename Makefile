.PHONY: install scrape mine personas review calibrate test clean

install:
	pip install -e .

scrape:
	paper-reviewer scrape

mine:
	paper-reviewer mine --abstract abstract.txt

personas:
	paper-reviewer personas

review:
	paper-reviewer review --pdf paper.pdf

calibrate:
	paper-reviewer calibrate --n 50

test:
	pytest tests/ -q

clean:
	rm -rf data/cache data/embeddings
