LANGUAGES = ar en es et fi fr pl ru zh
BASE_URL = https://dl.fbaipublicfiles.com/fasttext/vectors-crawl
FILES = $(foreach lang,$(LANGUAGES),cc.$(lang).300.vec)

all: $(FILES)

cc.%.300.vec:
	@echo "Downloading $@..."
	@wget $(BASE_URL)/$@.gz
	@gzip -d $@.gz

install:
	@pip install pyconcepticon; pip install numpy

clean:
	@echo "Cleaning up downloaded files..."
	@rm -f cc.*.*.vec

.PHONY: all install clean