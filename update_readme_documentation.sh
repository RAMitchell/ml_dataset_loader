#!/bin/sh

python -m pydocmd simple datasets+ > doc.md

doc_start="\[comment\]: # (Begin generated documentation)"
doc_end="\[comment\]: # (End generated documentation)"
sed -i -n "/${doc_start}/{p;:a;N;/${doc_end}/!ba;s/.*\n//};p" README.md
sed -i "/${doc_start}/ r doc.md" README.md

rm doc.md
