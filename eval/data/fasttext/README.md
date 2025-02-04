# Mapping fastText embeddings onto Concepticon's edition of Multi-SimLex

```bash
make install  # installs necessary Python packages (numpy, pyconcepticon)
make  # downloads fastText vectors
python preprocess.py  # reduce fastText vectors to 128 dimensions
python map_ft_embeddings.py  # run the Python script to map vectors onto their corresponding entries
make clean  # remove fastText vectors from disk
```
