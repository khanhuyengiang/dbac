apidoc:
	sphinx-apidoc -f -M -e -T -o api ../src/dbac

html: apidoc
	sphinx-build -b html . _build/html

clean:
	rm -rf _build
