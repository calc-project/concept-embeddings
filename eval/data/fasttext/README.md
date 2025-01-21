# Mapping fastText embeddings onto Concepticon's edition of Multi-SimLex

```bash
make install  # installs necessary Python packages (numpy, pyconcepticon)
make  # downloads fastText vectors
python map_ft_embeddings.py  # run the Python script to map vectors onto their corresponding entries
make clean  # remove fastText vectors from disk
```
