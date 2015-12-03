.PHONY: allprogress cleanprogress allfinal cleanfinal

cleanprogress:
	rm -f progress.pdf

allprogress: progress.md
	pandoc -t beamer -s progress.md -o progress.pdf

cleanfinal:
	rm -f final.pdf

allfinal: final.md
	pandoc -t beamer -s final.md -o final.pdf
