.PHONY: progress final clean

progress: progress.md
	pandoc -t beamer -s progress.md -o progress.pdf

final: final.md
	pandoc -t beamer -s final.md -o final.pdf

clean:
	rm -f progress.pdf
	rm -f final.pdf
	
