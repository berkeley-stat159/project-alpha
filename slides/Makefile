.PHONY: allprogress cleanprogress allfinal cleanfinal

allprogress: progress.md
	pandoc -t beamer -s progress.md -o progress.pdf

cleanprogress:
	rm -f progress.pdf

allfinal: final.md
	pandoc -t beamer -s final.md -o final.pdf

cleanfinal:
	rm -f final.pdf
