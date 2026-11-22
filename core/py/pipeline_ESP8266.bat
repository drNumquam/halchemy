python pdf2json.py --pdf ./doc/0a-esp8266ex_datasheet_en.pdf --output ./ESP8266
python toc_ranker.py --toc ./ESP8266/toc.txt --prompt peripheral_prompt.txt --output ./ESP8266 --concurrent 2