.PHONY: all clean

clean:
	rm -f progress.pdf

all: progress.md
	pandoc -t beamer -s progress.md -o progress.pdf
